// ==UserScript==
// @name         iLabel SPA双线程防碰撞 (测试版)
// @namespace    http://tampermonkey.net/
// @version      3.2
// @description  解决并发问题，完美适配Ant Design弹窗，支持手动清空黑板状态，并推送数据到本地API
// @author       Your Expert Assistant
// @match        *://ilabel.alipay.com/*
// @match        file:///*
// @grant        GM_xmlhttpRequest
// ==/UserScript==

(function() {
    'use strict';

    // ==========================================
    // 1. 身份初始化
    // ==========================================
    const tabId = 'Tab_' + Math.random().toString(36).substring(2, 6).toUpperCase();
    console.log(`🚀 [防碰撞助手] 启动！当前标签页ID: ${tabId}`);

    let lastProcessedTaskId = null;
    let consecutiveSkips = 0;
    let isPaused = false;
    let isSkipping = false;

    // ==========================================
    // 2. 可视化 Debug 悬浮窗
    // ==========================================
    const debugPanel = document.createElement('div');
    // 注意：pointer-events 改为 auto，以便点击按钮
    debugPanel.style.cssText = `
        position: fixed; bottom: 20px; left: 20px;
        background: rgba(0, 0, 0, 0.85); color: #00FF00;
        padding: 15px; border-radius: 8px; font-family: monospace;
        font-size: 13px; z-index: 999999; pointer-events: auto;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5); min-width: 300px;
        border: 1px solid #444;
    `;
    document.body.appendChild(debugPanel);

    // 定义清空黑板的方法
    window.clearIlabelBlackboard = function() {
        localStorage.setItem('ilabel_active_tasks', '{}');
        console.log("🧹 [防碰撞助手] 黑板已手动清空");
        // 视觉反馈
        debugPanel.style.borderColor = "#FFF";
        setTimeout(() => { debugPanel.style.borderColor = "#444"; }, 500);
    };

    function updateDebugUI(currentScreenTaskId, statusMsg = "") {
        let activeTasks = JSON.parse(localStorage.getItem('ilabel_active_tasks') || '{}');

        let html = `<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <strong style="color:white; font-size:14px;">🛠️ 黑板状态 (我是 ${tabId})</strong>
            <button onclick="window.clearIlabelBlackboard()" style="cursor:pointer; background:#f5222d; color:white; border:none; padding:2px 8px; border-radius:4px; font-size:11px;">清空黑板</button>
        </div><hr style="border-color:#444; margin:5px 0;">`;

        html += `<span style="color:#FFF">当前屏幕UUID:</span> <span style="color:yellow">${currentScreenTaskId ? currentScreenTaskId.substring(0,8)+'...' : '空(加载中)'}</span><br><br>`;

        let taskEntries = Object.keys(activeTasks);
        if (taskEntries.length === 0) {
            html += `<span style="color:#888">黑板是空的</span><br>`;
        } else {
            for (let key of taskEntries) {
                let isMe = (key === tabId);
                let color = isMe ? '#00FF00' : '#00BFFF';
                let shortTask = activeTasks[key] ? activeTasks[key].substring(0, 8) + '...' : '空';
                html += `<span style="color:${color}">${key}: ${shortTask} ${isMe ? '(本机)' : ''}</span><br>`;
            }
        }

        if (statusMsg) html += `<br><span style="color:#FF00FF">▶ ${statusMsg}</span>`;

        debugPanel.innerHTML = html;
    }

    // ==========================================
    // 3. 提取 UUID 及结果轮询展示
    // ==========================================
    let resultPollingTimer = null;
    let resultDisplayed = false;

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

    function startResultPolling(taskId) {
        if (resultPollingTimer) clearInterval(resultPollingTimer);
        resultDisplayed = false;
        
        resultPollingTimer = setInterval(() => {
            if (!taskId || resultDisplayed) {
                clearInterval(resultPollingTimer);
                return;
            }
            
            GM_xmlhttpRequest({
                method: "GET",
                url: `http://127.0.0.1:8001/api/task_result/${taskId}`,
                onload: function(response) {
                    try {
                        let res = JSON.parse(response.responseText);
                        if (res.status === "finished" && res.data) {
                            displayResult(res.data, taskId);
                            resultDisplayed = true;
                            clearInterval(resultPollingTimer);
                        }
                    } catch(e) {
                        console.error("解析结果失败", e);
                    }
                }
            });
        }, 3000);
    }

    function displayResult(data, taskId) {
        let taskInfo = getTaskUUIDElement();
        if (!taskInfo || taskInfo.text !== taskId) return;
        
        let rect = taskInfo.el.getBoundingClientRect();
        
        let resultDiv = document.createElement("div");
        resultDiv.id = "langgraph-result-" + taskId;
        resultDiv.style.cssText = `
            position: absolute;
            top: ${rect.bottom + window.scrollY}px;
            left: ${rect.right + window.scrollX + 10}px;
            background: #f6ffed;
            border: 1px solid #b7eb8f;
            color: #389e0d;
            padding: 10px;
            border-radius: 6px;
            font-size: 13px;
            z-index: 1000000;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            max-width: 350px;
            word-wrap: break-word;
            line-height: 1.5;
        `;
        
        let content = `<strong style="color:#096dd9">[LangGraph WebUI 结果]</strong><br>`;
        content += `<b>层级:</b> ${data.hierarchy || '无'} &nbsp;|&nbsp; <b>难度:</b> ${data.difficulty || '无'}<br>`;
        content += `<b>类型:</b> ${data.problem_type || '无'}<br>`;
        if (data.trap_analysis) {
             content += `<span style="color:#f5222d">⚠️ <b>陷阱:</b> ${data.trap_reason}</span><br>`;
        }
        if (data.final_decision) {
             let color = data.final_decision.includes('Match') ? '#389e0d' : '#f5222d';
             content += `<span style="color:${color}"><b>裁判结论:</b> ${data.final_decision}</span> (置信度: ${data.confidence_score})<br>`;
        }
        
        resultDiv.innerHTML = content;
        
        let old = document.getElementById("langgraph-result-" + taskId);
        if (old) old.remove();
        
        document.body.appendChild(resultDiv);
    }

    // ==========================================
    // 3.5 提取题目和答案并发送到服务端
    // ==========================================
    let hasSentData = false;

    async function extractAndSendData(taskId) {
        if (hasSentData) return;

        // 提取题目和答案：使用网页中的“复制原文本”按钮
        let questionContent = "";
        let answerContent = "";

        let copyBtns = Array.from(document.querySelectorAll('button')).filter(b => b.innerText.includes('复制原文本'));
        if (copyBtns.length >= 2) {
            // 定义异步拦截剪贴板方法，防止阻塞
            const fetchClipboardDataFromBtn = async (btn) => {
                return new Promise((resolve) => {
                    let tempText = "";

                    // 拦截 navigator.clipboard.writeText
                    let originalWriteText = navigator.clipboard ? navigator.clipboard.writeText : null;
                    if (originalWriteText) {
                        navigator.clipboard.writeText = function(text) {
                            tempText = text;
                            return Promise.resolve(); // 阻止真正写入系统的剪贴板，以免影响用户
                        };
                    }

                    // 拦截 document.execCommand('copy') 事件
                    let copyHandler = function(e) {
                        let originalSetData = e.clipboardData.setData;
                        e.clipboardData.setData = function(format, data) {
                            if (format === 'text/plain' || format === 'text') tempText = data;
                            return originalSetData.call(e.clipboardData, format, data);
                        };
                    };
                    document.addEventListener('copy', copyHandler);

                    // 模拟点击提取内容
                    btn.click();

                    // 等待页面原本的回调（如 React state、setTimeout 或微任务跑完）
                    setTimeout(() => {
                        if (originalWriteText) navigator.clipboard.writeText = originalWriteText;
                        document.removeEventListener('copy', copyHandler);
                        resolve(tempText);
                    }, 150);
                });
            };

            questionContent = await fetchClipboardDataFromBtn(copyBtns[0]);
            answerContent = await fetchClipboardDataFromBtn(copyBtns[1]);
        } else {
            // 降级兼容：如果在某些界面找不到对应按钮，使用原来的方法提取
            let qNode = document.evaluate('//div[contains(text(),"question")]/following-sibling::div | //div[contains(text(),"题目内容")]/following-sibling::div', document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
            if(qNode) questionContent = qNode.innerText.trim();

            let aNode = document.evaluate('//div[contains(text(),"answer")]/following-sibling::div | //div[contains(text(),"平台给出的参考答案")]/following-sibling::div | //div[contains(text(),"参考答案")]/following-sibling::div', document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
            if(aNode) answerContent = aNode.innerText.trim();

            if (!questionContent || !answerContent) {
               let allDivs = document.querySelectorAll('div');
               for(let i=0; i<allDivs.length; i++) {
                   let text = allDivs[i].innerText.trim();
                   if(text.includes('question') || text.includes('题目内容')) {
                       let nextDiv = allDivs[i].nextElementSibling;
                       if (nextDiv && !questionContent) questionContent = nextDiv.innerText.trim();
                   }
                   if(text.includes('answer') || text.includes('平台给出的参考答案')) {
                       let nextDiv = allDivs[i].nextElementSibling;
                       if (nextDiv && !answerContent) answerContent = nextDiv.innerText.trim();
                   }
               }
            }
        }

        if (questionContent || answerContent) {
            console.log("📝 提取到题目与答案，准备发送...");
            GM_xmlhttpRequest({
                method: "POST",
                url: "http://127.0.0.1:8001/api/task_data",
                data: JSON.stringify({
                    "task_id": taskId || "unknown",
                    "question_content": questionContent,
                    "answer": answerContent
                }),
                headers: {
                    "Content-Type": "application/json"
                },
                onload: function(response) {
                    console.log("✅ 发送数据成功:", response.responseText);
                    hasSentData = true; // 当前页面/任务只发一次
                },
                onerror: function(err) {
                    console.error("❌ 发送数据失败:", err);
                }
            });
        }
    }

    // ==========================================
    // 4. 操作黑板
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
    // 5. 核心跳过自动化流程
    // ==========================================
    function executeSkipSequence() {
        if (isSkipping) return;
        isSkipping = true;

        consecutiveSkips++;
        if (consecutiveSkips >= 5) {
            isPaused = true;
            isSkipping = false;
            alert("防碰撞助手：连续跳过同一任务5次！为防异常，脚本已暂停。");
            return;
        }

        updateDebugUI(lastProcessedTaskId, "正在查找跳过按钮...");

        let buttons = Array.from(document.querySelectorAll('button'));
        let skipBtn = buttons.find(btn => btn.innerText.replace(/\s+/g, '') === '跳过');

        if (!skipBtn) {
            console.error("❌ 找不到跳过按钮");
            isSkipping = false;
            return;
        }

        updateDebugUI(lastProcessedTaskId, "已点击跳过，等待弹窗...");
        skipBtn.click();

        setTimeout(() => {
            let textarea = document.getElementById('reason');
            if (textarea) {
                updateDebugUI(lastProcessedTaskId, "正在填写理由...");

                let nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
                nativeInputValueSetter.call(textarea, "双线程自动防碰撞跳过");
                textarea.dispatchEvent(new Event('input', { bubbles: true }));

                setTimeout(() => {
                    updateDebugUI(lastProcessedTaskId, "点击确定...");
                    let footerBtns = Array.from(document.querySelectorAll('.ant-modal-footer button, div[style*="flex-end"] button'));
                    let confirmBtn = footerBtns.find(btn => btn.innerText.replace(/\s+/g, '') === '跳过');

                    if (confirmBtn) {
                        confirmBtn.click();
                        console.log("✅ 自动跳过流程完成！");
                        setTimeout(() => { isSkipping = false; }, 1000);
                    } else {
                        console.error("❌ 找不到确定按钮");
                        isSkipping = false;
                    }
                }, 300);

            } else {
                console.error("❌ 找不到理由输入框(id=reason)");
                isSkipping = false;
            }
        }, 600);
    }


    // ==========================================
    // 6. 核心引擎
    // ==========================================
    function runStateMachine() {
        if (isPaused) return;

        let currentScreenTaskId = getTaskUUID();
        // 只有在非跳过过程中才更新通用UI
        if (!isSkipping) updateDebugUI(currentScreenTaskId);

        if (currentScreenTaskId && currentScreenTaskId !== lastProcessedTaskId) {
            if (lastProcessedTaskId) releaseTask();

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
                debugPanel.style.backgroundColor = "rgba(255, 0, 0, 0.85)";
                setTimeout(() => debugPanel.style.backgroundColor = "rgba(0, 0, 0, 0.85)", 300);
                executeSkipSequence();
            } else {
                registerTask(currentScreenTaskId);
                consecutiveSkips = 0;

                // 当前任务没有碰撞，或者是本地测试中，尝试提取数据
                extractAndSendData(currentScreenTaskId);
                startResultPolling(currentScreenTaskId);
            }
        }
        else if (!currentScreenTaskId && lastProcessedTaskId && !isSkipping) {
            // Task has been removed/completed
            console.log(`🗑️ 任务完成或切题，清空任务: ${lastProcessedTaskId}`);
            if (resultPollingTimer) clearInterval(resultPollingTimer);
            GM_xmlhttpRequest({
                method: "POST",
                url: "http://127.0.0.1:8001/api/cancel_task",
                data: JSON.stringify({
                    "task_id": lastProcessedTaskId
                }),
                headers: { "Content-Type": "application/json" },
                onload: function(res) { console.log("✅ 发送清空命令成功") }
            });

            releaseTask();
            lastProcessedTaskId = null;
            hasSentData = false; // 用户切题后清除标记
        }

        let jitterDelay = 400 + Math.random() * 300;
        setTimeout(runStateMachine, jitterDelay);
    }

    runStateMachine();

    window.addEventListener('beforeunload', releaseTask);

})();