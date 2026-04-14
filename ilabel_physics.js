// ==UserScript==
// @name         iLabel 物理评分助手
// @namespace    http://tampermonkey.net/
// @version      1.0
// @description  自动提取物理题的marking标准和模型回答，调用本地API评分，并在页面上展示得分要点和分值
// @author       Your Expert Assistant
// @match        *://ilabel.alipay.com/*
// @match        file:///*
// @grant        GM_xmlhttpRequest
// @grant        GM.xmlHttpRequest
// @connect      127.0.0.1
// @connect      localhost
// ==/UserScript==

(function() {
    'use strict';

    // ==========================================
    // 1. 身份初始化
    // ==========================================
    const tabId = 'PhysTab_' + Math.random().toString(36).substring(2, 6).toUpperCase();
    console.log(`🔬 [物理评分助手] 启动！当前标签页ID: ${tabId}`);

    let lastProcessedTaskId = null;
    let isSkipping = false;
    let isPaused = false;
    let consecutiveSkips = 0;
    let hasSentData = false;

    // ==========================================
    // 2. 可视化 Debug 悬浮窗（左下角）
    // ==========================================
    const debugPanel = document.createElement('div');
    debugPanel.style.cssText = `
        position: fixed; bottom: 20px; left: 20px;
        background: rgba(10, 20, 50, 0.90); color: #7DF9FF;
        padding: 12px 15px; border-radius: 8px; font-family: monospace;
        font-size: 12px; z-index: 999999; pointer-events: auto;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5); min-width: 270px;
        border: 1px solid #1a3a6a;
    `;
    document.body.appendChild(debugPanel);

    unsafeWindow.clearIlabelBlackboard = function() {
        localStorage.setItem('ilabel_active_tasks', '{}');
        debugPanel.style.borderColor = "#FFF";
        setTimeout(() => { debugPanel.style.borderColor = "#1a3a6a"; }, 500);
    };

    function updateDebugUI(currentScreenTaskId, statusMsg = "") {
        let activeTasks = JSON.parse(localStorage.getItem('ilabel_active_tasks') || '{}');
        let html = `<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <strong style="color:white; font-size:13px;">🔬 物理评分助手 (${tabId})</strong>
            <button onclick="window.clearIlabelBlackboard()" style="cursor:pointer; background:#c0392b; color:white; border:none; padding:2px 7px; border-radius:4px; font-size:11px;">清空黑板</button>
        </div><hr style="border-color:#1a3a6a; margin:4px 0;">`;

        html += `<span style="color:#AAA">当前UUID:</span> <span style="color:yellow">${currentScreenTaskId ? currentScreenTaskId.substring(0,8)+'...' : '加载中'}</span><br>`;

        let taskEntries = Object.keys(activeTasks);
        if (taskEntries.length === 0) {
            html += `<span style="color:#555">黑板为空</span>`;
        } else {
            for (let key of taskEntries) {
                let isMe = (key === tabId);
                let color = isMe ? '#7DF9FF' : '#00BFFF';
                let shortTask = activeTasks[key] ? activeTasks[key].substring(0, 8) + '...' : '空';
                html += `<span style="color:${color}">${key}: ${shortTask}${isMe ? ' (本机)' : ''}</span><br>`;
            }
        }
        if (statusMsg) html += `<br><span style="color:#FF9F43">▶ ${statusMsg}</span>`;
        debugPanel.innerHTML = html;
    }

    // ==========================================
    // 3. 结果展示窗口（右侧）
    // ==========================================
    let resultPollingTimer = null;
    let resultDisplayed = false;

    function createResultWindow() {
        let resultDiv = document.getElementById("physics-result-window");
        if (!resultDiv) {
            resultDiv = document.createElement("div");
            resultDiv.id = "physics-result-window";
            resultDiv.style.cssText = `
                position: fixed; top: 20px; right: 20px;
                background: rgba(10, 20, 50, 0.92); color: #E0E0E0;
                padding: 15px; border-radius: 8px; font-family: monospace;
                font-size: 12px; z-index: 1000000; pointer-events: auto;
                box-shadow: 0 4px 16px rgba(0,0,0,0.6); min-width: 340px; max-width: 460px;
                border: 1px solid #1a3a6a; word-wrap: break-word; line-height: 1.6;
                max-height: 80vh; overflow-y: auto;
            `;
            document.body.appendChild(resultDiv);
        }
        return resultDiv;
    }

    function displayPendingState(taskId) {
        let resultDiv = createResultWindow();
        resultDiv.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <strong style="color:white; font-size:13px;">🔬 物理评分结果</strong>
            </div>
            <hr style="border-color:#1a3a6a; margin:5px 0;">
            <span style="color:#F9CA24">⏳ 正在调用 Flash 模型评分，请稍候...</span>
        `;
        resultDiv.dataset.taskId = taskId;
    }

    function displayResult(data, taskId) {
        let taskInfo = getTaskUUIDElement();
        if (!taskInfo || taskInfo.text !== taskId) return;

        let resultDiv = createResultWindow();
        let results = data.results || [];
        let scoreString = data.score_string || "0";
        let totalScore = data.total_score !== undefined ? data.total_score : 0;

        let html = `<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <strong style="color:white; font-size:13px;">🔬 物理评分结果</strong>
        </div>
        <hr style="border-color:#1a3a6a; margin:5px 0;">`;

        // 分值总览（醒目显示）
        html += `<div style="background:#162040; border-radius:5px; padding:8px 10px; margin-bottom:10px; text-align:center;">
            <span style="color:#AAA; font-size:11px;">总分</span><br>
            <span style="color:#F9CA24; font-size:18px; font-weight:bold;">${scoreString}</span>
        </div>`;

        // 逐条得分要点
        html += `<div style="color:#7DF9FF; font-size:11px; margin-bottom:5px; font-weight:bold;">得分要点</div>`;
        for (let i = 0; i < results.length; i++) {
            let r = results[i];
            let awarded = r.points_awarded !== undefined ? r.points_awarded : 0;
            let maxPts = r.max_points !== undefined ? r.max_points : 0;
            let satisfied = r.satisfied === true || awarded > 0;
            let icon = satisfied ? '✓' : '✗';
            let iconColor = satisfied ? '#2ecc71' : '#e74c3c';
            let ptColor = satisfied ? '#2ecc71' : '#e74c3c';

            // 截断 criterion 文本，只显示要点
            let criterionShort = (r.criterion || '').length > 80
                ? (r.criterion || '').substring(0, 80) + '...'
                : (r.criterion || '');

            html += `<div style="margin-bottom:8px; padding:6px 8px; background:#0d1a35; border-radius:4px; border-left:3px solid ${iconColor};">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <span style="color:${iconColor}; margin-right:6px; flex-shrink:0;">${icon}</span>
                    <span style="color:#CCC; flex:1; font-size:11px;">${criterionShort}</span>
                    <span style="color:${ptColor}; font-weight:bold; margin-left:8px; flex-shrink:0;">${awarded}分</span>
                </div>`;

            if (r.reason) {
                let reasonShort = r.reason.length > 100 ? r.reason.substring(0, 100) + '...' : r.reason;
                html += `<div style="color:#888; font-size:10px; margin-top:3px; margin-left:16px;">${reasonShort}</div>`;
            }
            html += `</div>`;
        }

        resultDiv.innerHTML = html;
        resultDiv.dataset.taskId = taskId;
    }

    function displayError(taskId, errorMsg) {
        let resultDiv = createResultWindow();
        resultDiv.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <strong style="color:white; font-size:13px;">🔬 物理评分结果</strong>
            </div>
            <hr style="border-color:#1a3a6a; margin:5px 0;">
            <span style="color:#e74c3c;">❌ 评分出错: ${errorMsg || '未知错误'}</span>
        `;
    }

    function startResultPolling(taskId) {
        if (resultPollingTimer) clearInterval(resultPollingTimer);
        resultDisplayed = false;
        displayPendingState(taskId);

        resultPollingTimer = setInterval(() => {
            if (!taskId || resultDisplayed) {
                clearInterval(resultPollingTimer);
                return;
            }
            GM_xmlhttpRequest({
                method: "GET",
                url: `http://localhost:8001/api/physics_result/${taskId}`,
                onload: function(response) {
                    try {
                        let res = JSON.parse(response.responseText);
                        if (res.status === "finished" && res.data) {
                            displayResult(res.data, taskId);
                            resultDisplayed = true;
                            clearInterval(resultPollingTimer);
                        } else if (res.status === "error") {
                            displayError(taskId, res.error);
                            resultDisplayed = true;
                            clearInterval(resultPollingTimer);
                        }
                    } catch(e) {
                        console.error("解析评分结果失败", e);
                    }
                }
            });
        }, 2500);
    }

    // ==========================================
    // 4. 提取 UUID
    // ==========================================
    function getTaskUUIDElement() {
        const uuidRegex = /^[a-f0-9]{32}$/i;
        let candidates = document.querySelectorAll('div[class*="textSrc"] > div');
        for (let el of candidates) {
            let text = el.innerText.trim();
            if (uuidRegex.test(text)) return { text: text, el: el };
        }
        let walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null, false);
        let node;
        while (node = walker.nextNode()) {
            let text = node.nodeValue.trim();
            if (uuidRegex.test(text)) return { text: text, el: node.parentElement };
        }
        return null;
    }

    function getTaskUUID() {
        let taskInfo = getTaskUUIDElement();
        return taskInfo ? taskInfo.text : null;
    }

    // ==========================================
    // 5. 提取 marking 评分标准 和 模型回答
    // ==========================================

    /**
     * 通用策略：遍历所有文本节点，找到包含目标标签的节点，
     * 然后从其父元素或相邻容器中提取内容。
     * 支持多种可能的 DOM 结构。
     */
    function extractFieldContent(labelText) {
        // 策略1：通过 "复制原文本" 按钮（最可靠，若有多个按钮时需要配对）
        // 本函数只做通用文本提取，按钮法由外部逻辑处理

        // 策略2：遍历 DOM 查找标签文本
        let allElements = document.querySelectorAll('*');
        for (let el of allElements) {
            // 只看叶子级或接近叶子级的元素（直接包含标签文本）
            if (el.children.length > 5) continue;
            let ownText = '';
            for (let child of el.childNodes) {
                if (child.nodeType === Node.TEXT_NODE) {
                    ownText += child.nodeValue;
                }
            }
            ownText = ownText.trim();
            if (ownText.toLowerCase().includes(labelText.toLowerCase())) {
                // 找到包含标签的元素，尝试获取其兄弟或父级的下一个兄弟
                let targets = [
                    el.nextElementSibling,
                    el.parentElement && el.parentElement.nextElementSibling,
                    el.closest('[class]') && el.closest('[class]').nextElementSibling,
                ];
                for (let target of targets) {
                    if (target && target.innerText && target.innerText.trim().length > 5) {
                        return target.innerText.trim();
                    }
                }
                // 尝试 XPath：找到下一个非空文本块
                let xpathResult = document.evaluate(
                    `following-sibling::*[normalize-space(.)!=''][1]`,
                    el, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null
                ).singleNodeValue;
                if (xpathResult && xpathResult.innerText) {
                    return xpathResult.innerText.trim();
                }
            }
        }

        // 策略3：TreeWalker 全文搜索
        let walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null, false);
        let node;
        while (node = walker.nextNode()) {
            let text = node.nodeValue.trim();
            if (text.toLowerCase().includes(labelText.toLowerCase())) {
                // 从这个文本节点往上找容器，再取下一个兄弟
                let parent = node.parentElement;
                for (let depth = 0; depth < 4; depth++) {
                    if (!parent) break;
                    let sibling = parent.nextElementSibling;
                    if (sibling && sibling.innerText && sibling.innerText.trim().length > 5) {
                        return sibling.innerText.trim();
                    }
                    parent = parent.parentElement;
                }
            }
        }
        return null;
    }

    /**
     * 专门提取 marking 字段。marking 字段通常是 JSON 数组字符串，
     * 也可能是原始的 [["..."]] 格式。
     */
    function extractMarkingContent() {
        let raw = extractFieldContent("marking:");
        if (!raw) raw = extractFieldContent("marking :");
        if (!raw) raw = extractFieldContent("Marking:");

        if (!raw) {
            // 尝试在全页面文本中找到 [[ 开头的 JSON 数组
            let bodyText = document.body.innerText;
            let jsonMatch = bodyText.match(/\[\s*\[.*?\]\s*\]/s);
            if (jsonMatch) raw = jsonMatch[0];
        }
        return raw ? raw.trim() : null;
    }

    /**
     * 专门提取模型回答字段。
     */
    function extractModelAnswer() {
        let raw = extractFieldContent("模型回答渲染:");
        if (!raw) raw = extractFieldContent("模型回答渲染 :");
        if (!raw) raw = extractFieldContent("模型回答:");
        return raw ? raw.trim() : null;
    }

    // ==========================================
    // 6. 发送物理评分请求
    // ==========================================
    async function extractAndSendPhysicsData(taskId) {
        if (hasSentData) return;

        // 尝试通过复制按钮提取（如果有多个按钮，第1个=marking，第2个=模型回答）
        let markingContent = null;
        let modelAnswerContent = null;

        let copyBtns = Array.from(document.querySelectorAll('button')).filter(b => b.innerText.includes('复制原文本'));

        if (copyBtns.length >= 2) {
            const fetchFromBtn = async (btn) => {
                return new Promise((resolve) => {
                    let tempText = "";
                    let originalWriteText = navigator.clipboard ? navigator.clipboard.writeText : null;
                    if (originalWriteText) {
                        navigator.clipboard.writeText = function(text) {
                            tempText = text;
                            return Promise.resolve();
                        };
                    }
                    let copyHandler = function(e) {
                        let originalSetData = e.clipboardData.setData;
                        e.clipboardData.setData = function(format, data) {
                            if (format === 'text/plain' || format === 'text') tempText = data;
                            return originalSetData.call(e.clipboardData, format, data);
                        };
                    };
                    document.addEventListener('copy', copyHandler);
                    btn.click();
                    setTimeout(() => {
                        if (originalWriteText) navigator.clipboard.writeText = originalWriteText;
                        document.removeEventListener('copy', copyHandler);
                        resolve(tempText);
                    }, 150);
                });
            };

            // 扫描按钮对应的标签，找到 marking 和 模型回答 的按钮
            for (let i = 0; i < copyBtns.length; i++) {
                let btn = copyBtns[i];
                // 查找此按钮附近的标签文本
                let container = btn.closest('[class]') || btn.parentElement;
                let containerText = container ? container.innerText : '';
                // 向上找几层
                for (let depth = 0; depth < 5 && container; depth++) {
                    containerText = container.innerText || '';
                    if (containerText.toLowerCase().includes('marking')) {
                        markingContent = await fetchFromBtn(btn);
                        break;
                    }
                    if (containerText.includes('模型回答渲染') || containerText.includes('模型回答')) {
                        modelAnswerContent = await fetchFromBtn(btn);
                        break;
                    }
                    container = container.parentElement;
                }
            }

            // 如果按钮分析没成功，降级到顺序匹配（第1=marking，倒数1=模型回答）
            if (!markingContent) markingContent = await fetchFromBtn(copyBtns[0]);
            if (!modelAnswerContent && copyBtns.length >= 2) {
                modelAnswerContent = await fetchFromBtn(copyBtns[copyBtns.length - 1]);
            }
        }

        // 降级：DOM 文本搜索
        if (!markingContent) markingContent = extractMarkingContent();
        if (!modelAnswerContent) modelAnswerContent = extractModelAnswer();

        if (!markingContent && !modelAnswerContent) {
            console.warn("⚠️ [物理评分助手] 未能提取到 marking 或模型回答，跳过发送");
            updateDebugUI(taskId, "未能提取字段，等待页面加载");
            return;
        }

        console.log("📐 [物理评分助手] 提取到数据，准备发送评分请求...");
        updateDebugUI(taskId, "正在发送评分请求...");

        GM_xmlhttpRequest({
            method: "POST",
            url: "http://localhost:8001/api/physics_task",
            data: JSON.stringify({
                "task_id": taskId || "unknown",
                "marking": markingContent || "[]",
                "model_answer": modelAnswerContent || ""
            }),
            headers: { "Content-Type": "application/json" },
            onload: function(response) {
                console.log("✅ [物理评分助手] 发送成功:", response.responseText);
                hasSentData = true;
                startResultPolling(taskId);
            },
            onerror: function(err) {
                console.error("❌ [物理评分助手] 发送失败:", err);
                updateDebugUI(taskId, "❌ 发送失败，检查本地服务是否运行");
            }
        });
    }

    // ==========================================
    // 7. 黑板操作（防碰撞）
    // ==========================================
    function registerTask(taskId) {
        let activeTasks = JSON.parse(localStorage.getItem('ilabel_active_tasks') || '{}');
        activeTasks[tabId] = taskId;
        localStorage.setItem('ilabel_active_tasks', JSON.stringify(activeTasks));
    }

    function releaseTask() {
        let activeTasks = JSON.parse(localStorage.getItem('ilabel_active_tasks') || '{}');
        if (activeTasks[tabId]) {
            delete activeTasks[tabId];
            localStorage.setItem('ilabel_active_tasks', JSON.stringify(activeTasks));
        }
    }

    // ==========================================
    // 8. 跳过序列（防碰撞）
    // ==========================================
    function executeSkipSequence() {
        if (isSkipping) return;
        isSkipping = true;
        consecutiveSkips++;
        if (consecutiveSkips >= 5) {
            isPaused = true;
            isSkipping = false;
            alert("物理评分助手：连续跳过同一任务5次！为防异常，脚本已暂停。");
            return;
        }
        updateDebugUI(lastProcessedTaskId, "正在查找跳过按钮...");
        let buttons = Array.from(document.querySelectorAll('button'));
        let skipBtn = buttons.find(btn => btn.innerText.replace(/\s+/g, '') === '跳过');
        if (!skipBtn) { isSkipping = false; return; }

        updateDebugUI(lastProcessedTaskId, "已点击跳过，等待弹窗...");
        skipBtn.click();
        setTimeout(() => {
            let textarea = document.getElementById('reason');
            if (textarea) {
                let nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
                nativeInputValueSetter.call(textarea, "双线程自动防碰撞跳过");
                textarea.dispatchEvent(new Event('input', { bubbles: true }));
                setTimeout(() => {
                    let footerBtns = Array.from(document.querySelectorAll('.ant-modal-footer button, div[style*="flex-end"] button'));
                    let confirmBtn = footerBtns.find(btn => btn.innerText.replace(/\s+/g, '') === '跳过');
                    if (confirmBtn) {
                        confirmBtn.click();
                        setTimeout(() => { isSkipping = false; }, 1000);
                    } else {
                        isSkipping = false;
                    }
                }, 300);
            } else {
                isSkipping = false;
            }
        }, 600);
    }

    // ==========================================
    // 9. 核心状态机
    // ==========================================
    function runStateMachine() {
        if (isPaused) return;

        let currentScreenTaskId = getTaskUUID();
        if (!isSkipping) updateDebugUI(currentScreenTaskId);

        if (currentScreenTaskId && currentScreenTaskId !== lastProcessedTaskId) {
            // 切题：清理旧任务
            if (lastProcessedTaskId) {
                if (resultPollingTimer) clearInterval(resultPollingTimer);
                let resultDiv = document.getElementById("physics-result-window");
                if (resultDiv) resultDiv.remove();
                releaseTask();
                hasSentData = false;
                resultDisplayed = false;
            }

            lastProcessedTaskId = currentScreenTaskId;

            let activeTasks = JSON.parse(localStorage.getItem('ilabel_active_tasks') || '{}');
            let isCollision = false;
            for (let key in activeTasks) {
                if (key !== tabId && activeTasks[key] === currentScreenTaskId) {
                    isCollision = true;
                    break;
                }
            }

            if (isCollision) {
                debugPanel.style.backgroundColor = "rgba(200, 30, 30, 0.90)";
                setTimeout(() => debugPanel.style.backgroundColor = "rgba(10, 20, 50, 0.90)", 300);
                executeSkipSequence();
            } else {
                registerTask(currentScreenTaskId);
                consecutiveSkips = 0;
                // 等页面完全渲染后再提取（延迟1.5s）
                setTimeout(() => extractAndSendPhysicsData(currentScreenTaskId), 1500);
            }
        }

        let jitterDelay = 400 + Math.random() * 300;
        setTimeout(runStateMachine, jitterDelay);
    }

    runStateMachine();
    window.addEventListener('beforeunload', releaseTask);

})();
