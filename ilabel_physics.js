    // ==UserScript==
    // @name         iLabel Physics NER Assistant v2.2
    // @namespace    http://tampermonkey.net/
    // @version      2.2
    // @description  自动检测物理题子题切换，调用LLM获取标注建议，自动NER划词标注（选文本→双击→输入分值→关闭）并填写总分。除"下一题"和"提交"外全自动化。
    // @author       Assistant
    // @match        *://ilabel.alipay.com/*
    // @match        file:///*
    // @grant        GM_xmlhttpRequest
    // @grant        GM.xmlHttpRequest
    // @grant        unsafeWindow
    // @connect      127.0.0.1
    // @connect      localhost
    // ==/UserScript==

    (function () {
        'use strict';

        // ==========================================
        // 0. 覆盖 window.prompt：屏蔽 iLabel 的"Copy to clipboard: Ctrl+C, Enter"原生弹窗
        //    iLabel 在双击 NER 文本后会调用 window.prompt() 让用户复制，我们直接返回 defaultValue
        //    使其无感知关闭，从而让后续的标注弹窗正常出现。
        // ==========================================
        try {
            const _win = (typeof unsafeWindow !== 'undefined') ? unsafeWindow : window;
            const _origPrompt = _win.prompt.bind(_win);
            _win.prompt = function (msg, defaultValue) {
                if (/copy|clipboard|ctrl/i.test(msg || '')) {
                    console.log('[物理划词] 自动关闭 clipboard prompt');
                    return defaultValue !== undefined ? String(defaultValue) : '';
                }
                return _origPrompt(msg, defaultValue);
            };
        } catch (e) {
            console.warn('[物理划词] 无法覆盖 window.prompt:', e);
        }

        // ==========================================
        // 1. 全局状态
        // ==========================================
        const tabId = 'PhysNER_' + Math.random().toString(36).substring(2, 6).toUpperCase();
        console.log(`🔬 [物理划词助手 v2.1] 启动，TabID: ${tabId}`);

        let lastProcessedKey = null;
        let isProcessing = false;
        let subQObserver = null;

        // 返修模式全局状态（localStorage 跨标签页共享）
        const PHYS_REVISION_MODE_KEY  = 'ilabel_phys_revision_mode';
        const PHYS_REVISION_QUEUE_KEY = 'ilabel_phys_revision_queue';

        function getRevisionMode() {
            return _pageLS.getItem(PHYS_REVISION_MODE_KEY) === 'true';
        }
        function setRevisionMode(val) {
            _pageLS.setItem(PHYS_REVISION_MODE_KEY, val ? 'true' : 'false');
        }
        function getRevisionQueue() {
            try { return JSON.parse(_pageLS.getItem(PHYS_REVISION_QUEUE_KEY) || '[]'); }
            catch (_) { return []; }
        }
        function setRevisionQueue(arr) {
            _pageLS.setItem(PHYS_REVISION_QUEUE_KEY, JSON.stringify(arr));
        }
        function clearRevisionQueue() {
            _pageLS.setItem(PHYS_REVISION_QUEUE_KEY, '[]');
        }

        // ==========================================
        // 防碰撞黑板（localStorage 跨标签页共享）
        // ==========================================
        const PHYS_BOARD_KEY = 'ilabel_phys_active_tasks';
        // @grant unsafeWindow 时脚本在沙箱中运行，沙箱自身的 localStorage 与页面不共享；
        // 必须显式使用 unsafeWindow.localStorage 才能跨标签页同步。
        const _pageLS = (typeof unsafeWindow !== 'undefined' ? unsafeWindow : window).localStorage;

        function registerPhysTask(processKey) {
            const board = JSON.parse(_pageLS.getItem(PHYS_BOARD_KEY) || '{}');
            board[tabId] = processKey;
            _pageLS.setItem(PHYS_BOARD_KEY, JSON.stringify(board));
        }

        function releasePhysTask() {
            const board = JSON.parse(_pageLS.getItem(PHYS_BOARD_KEY) || '{}');
            if (board[tabId]) {
                delete board[tabId];
                _pageLS.setItem(PHYS_BOARD_KEY, JSON.stringify(board));
            }
        }

        /** 返回已占用此 processKey 的其他 tabId，无冲突返回 null */
        function checkPhysCollision(processKey) {
            const board = JSON.parse(_pageLS.getItem(PHYS_BOARD_KEY) || '{}');
            for (const key in board) {
                if (key !== tabId && board[key] === processKey) return key;
            }
            return null;
        }

        try {
            const _win = (typeof unsafeWindow !== 'undefined') ? unsafeWindow : window;
            _win.clearPhysBlackboard = function () {
                _win.localStorage.setItem(PHYS_BOARD_KEY, '{}');
                console.log('[物理划词] 黑板已手动清空');
            };
            _win.togglePhysRevisionMode = function () {
                const newMode = !getRevisionMode();
                setRevisionMode(newMode);
                clearRevisionQueue(); // 切换时清空队列
                log(newMode
                    ? '⚠ 返修模式已开启（标注失败的题目将加入队列）'
                    : '✅ 返修模式已关闭');
                updateStatus(newMode ? '返修模式: 开' : '返修模式: 关');
            };
            _win.sendPhysRevisionQueue = function () {
                const queue = getRevisionQueue();
                if (queue.length === 0) { log('返修队列为空'); return; }
                const count = queue.length;
                log(`正在发送 ${count} 条返修请求...`);
                GM_xmlhttpRequest({
                    method: 'POST',
                    url: 'http://localhost:8001/api/physics_revision',
                    headers: { 'Content-Type': 'application/json' },
                    data: JSON.stringify({ tasks: queue }),
                    onload(resp) {
                        try {
                            JSON.parse(resp.responseText);
                            clearRevisionQueue();
                            log(`✓ 已发送 ${count} 条返修请求，队列已清空`);
                            updateStatus('返修已发送');
                        } catch (e) {
                            log('返修发送失败: 响应解析错误');
                        }
                    },
                    onerror() { log('返修发送失败: 网络请求错误'); }
                });
            };
            _win.clearPhysRevisionQueue = function () {
                clearRevisionQueue();
                log('返修队列已清空');
                updateStatus('返修队列清空');
            };
        } catch (_) {}

        // ==========================================
        // 2. 工具函数
        // ==========================================
        function sleep(ms) {
            return new Promise(resolve => setTimeout(resolve, ms));
        }

        function log(msg) {
            console.log(`[物理划词] ${msg}`);
            updateStatus(msg);
        }

        // ── DEBUG 辅助（始终输出到控制台，不影响悬浮窗）──
        function dbg(tag, ...args) {
            console.log(`[物理DEBUG][${tag}]`, ...args);
        }
        function dbgWarn(tag, ...args) {
            console.warn(`[物理DEBUG][${tag}]`, ...args);
        }
        function dbgErr(tag, ...args) {
            console.error(`[物理DEBUG][${tag}]`, ...args);
        }

        // ==========================================
        // 3. 调试悬浮窗（左下角）
        // ==========================================
        const debugPanel = document.createElement('div');
        debugPanel.style.cssText = `
            position: fixed; bottom: 20px; left: 20px;
            background: rgba(10, 20, 50, 0.92); color: #7DF9FF;
            padding: 12px 15px; border-radius: 8px; font-family: monospace;
            font-size: 12px; z-index: 999999; pointer-events: auto;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5); min-width: 300px; max-width: 420px;
            border: 1px solid #1a3a6a; word-wrap: break-word;
        `;
        document.body.appendChild(debugPanel);

        function updateStatus(msg) {
            const uuid = getTaskUUID() || '—';
            const subQ = getSubQuestionId() || '—';
            const markIdx = getCurrentMarkItemIndex();

            // 黑板状态
            const board = JSON.parse(_pageLS.getItem(PHYS_BOARD_KEY) || '{}');
            const boardEntries = Object.entries(board);
            let boardHtml = '';
            if (boardEntries.length === 0) {
                boardHtml = '<span style="color:#888">黑板为空</span>';
            } else {
                boardHtml = boardEntries.map(([k, v]) => {
                    const isMe = k === tabId;
                    const color = isMe ? '#2ecc71' : '#00BFFF';
                    return `<span style="color:${color}">${k}: ${v} ${isMe ? '(本机)' : ''}</span>`;
                }).join('<br>');
            }

            const _revMode  = getRevisionMode();
            const _revQueue = getRevisionQueue();
            const revBtnStyle = _revMode
                ? 'cursor:pointer;background:#e67e22;color:white;border:none;padding:2px 8px;border-radius:4px;font-size:11px;'
                : 'cursor:pointer;background:#2c3e50;color:#ccc;border:1px solid #555;padding:2px 8px;border-radius:4px;font-size:11px;';
            const revSendHtml = (_revMode && _revQueue.length > 0) ? `
                <div style="margin-top:5px;display:flex;gap:4px;">
                    <button onclick="window.sendPhysRevisionQueue&&window.sendPhysRevisionQueue()"
                        style="cursor:pointer;background:#27ae60;color:white;border:none;padding:3px 0;border-radius:4px;font-size:11px;flex:1;">
                        📤 发送返修 (${_revQueue.length})
                    </button>
                    <button onclick="window.clearPhysRevisionQueue&&window.clearPhysRevisionQueue()"
                        style="cursor:pointer;background:#7f8c8d;color:white;border:none;padding:3px 6px;border-radius:4px;font-size:11px;">
                        清空
                    </button>
                </div>` : '';

            debugPanel.innerHTML = `
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                    <strong style="color:white;font-size:13px;">🔬 物理划词助手 (${tabId})</strong>
                    <button onclick="window.clearPhysBlackboard&&window.clearPhysBlackboard()" style="cursor:pointer;background:#f5222d;color:white;border:none;padding:2px 8px;border-radius:4px;font-size:11px;">清空黑板</button>
                </div>
                <hr style="border-color:#1a3a6a;margin:4px 0;">
                <span style="color:#AAA">UUID:</span> <span style="color:yellow">${uuid ? uuid.substring(0, 12) + '...' : '—'}</span><br>
                <span style="color:#AAA">子题:</span> <span style="color:#7DF9FF">${subQ || '—'}</span> &nbsp;
                <span style="color:#AAA">小题:</span> <span style="color:#FF9F43">${markIdx >= 0 ? markIdx : '—'}</span><br>
                <span style="color:#AAA">状态:</span> <span style="color:#2ecc71">${isProcessing ? '⏳ 处理中...' : '✅ 就绪'}</span><br>
                <hr style="border-color:#1a3a6a;margin:4px 0;">
                <span style="color:#AAA;font-size:11px;">黑板:</span><br>${boardHtml}<br>
                <span style="color:#FF9F43">▶ ${msg || ''}</span>
                <hr style="border-color:#1a3a6a;margin:4px 0;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <button onclick="window.togglePhysRevisionMode&&window.togglePhysRevisionMode()" style="${revBtnStyle}">
                        ${_revMode ? '⚠ 返修模式: 开' : '⚪ 返修模式: 关'}
                    </button>
                    ${_revMode ? `<span style="color:#FF9F43;font-size:11px;">队列: ${_revQueue.length} 项</span>` : ''}
                </div>
                ${revSendHtml}
            `;
        }

        // ==========================================
        // 4. 结果展示窗口（右上角）
        // ==========================================
        function createResultWindow() {
            let w = document.getElementById('phys-ner-result-window');
            if (!w) {
                w = document.createElement('div');
                w.id = 'phys-ner-result-window';
                w.style.cssText = `
                    position: fixed; bottom: 155px; left: 20px;
                    background: rgba(10, 20, 50, 0.93); color: #E0E0E0;
                    padding: 8px 12px; border-radius: 8px; font-family: monospace;
                    font-size: 11px; z-index: 1000000; pointer-events: auto;
                    box-shadow: 0 4px 16px rgba(0,0,0,0.6); min-width: 240px; max-width: 340px;
                    border: 1px solid #1a3a6a; word-wrap: break-word; line-height: 1.5;
                    max-height: 32vh; overflow-y: auto;
                `;
                document.body.appendChild(w);
            }
            return w;
        }

        function showResult(data, subQId) {
            const w = createResultWindow();
            const results = data.results || [];
            let html = `
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                    <strong style="color:white;font-size:13px;">🔬 划词标注结果</strong>
                    <span style="color:#888;font-size:11px;">${subQId || ''}</span>
                </div>
                <hr style="border-color:#1a3a6a;margin:5px 0;">
                <div style="background:#162040;border-radius:5px;padding:8px 10px;margin-bottom:10px;text-align:center;">
                    <span style="color:#AAA;font-size:11px;">总分</span><br>
                    <span style="color:#F9CA24;font-size:18px;font-weight:bold;">${data.score_string || '0'}</span>
                </div>
            `;
            html += `<div style="color:#7DF9FF;font-size:11px;margin-bottom:5px;font-weight:bold;">得分要点 (含标注情况)</div>`;
            for (let i = 0; i < results.length; i++) {
                const r = results[i];
                const satisfied = r.satisfied === true || r.points_awarded > 0;
                const icon = satisfied ? '✓' : '✗';
                const iconColor = satisfied ? '#2ecc71' : '#e74c3c';
                const criterionShort = (r.criterion || '').substring(0, 80) + ((r.criterion || '').length > 80 ? '...' : '');
                const matchedShort = r.matched_text ? `"${r.matched_text.substring(0, 50)}..."` : '(无匹配)';
                html += `
                    <div style="margin-bottom:8px;padding:6px 8px;background:#0d1a35;border-radius:4px;border-left:3px solid ${iconColor};">
                        <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                            <span style="color:${iconColor};margin-right:6px;flex-shrink:0;">${icon}</span>
                            <span style="color:#CCC;flex:1;font-size:11px;">${criterionShort}</span>
                            <span style="color:${iconColor};font-weight:bold;margin-left:8px;flex-shrink:0;">${r.points_awarded || 0}分</span>
                        </div>
                        ${satisfied ? `<div style="color:#888;font-size:10px;margin-top:3px;margin-left:16px;">🖊 ${matchedShort}</div>` : ''}
                    </div>
                `;
            }
            w.innerHTML = html;
        }

        function showPending(subQId) {
            const w = createResultWindow();
            w.innerHTML = `
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                    <strong style="color:white;font-size:13px;">🔬 划词标注结果</strong>
                    <span style="color:#888;font-size:11px;">${subQId || ''}</span>
                </div>
                <hr style="border-color:#1a3a6a;margin:5px 0;">
                <span style="color:#F9CA24">⏳ 正在调用AI评分，请稍候...</span>
            `;
        }

        // ==========================================
        // 5. DOM 提取函数
        // ==========================================

        function getTaskUUID() {
            const uuidRegex = /^[a-f0-9]{32}$/i;
            for (const el of document.querySelectorAll('[class*="textSrc"] > div')) {
                const t = el.innerText.trim();
                if (uuidRegex.test(t)) return t;
            }
            const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
            let node;
            while ((node = walker.nextNode())) {
                const t = node.nodeValue.trim();
                if (uuidRegex.test(t)) return t;
            }
            return null;
        }

        function getSubQuestionId() {
            // 优先：左上角"标注中(N / M)"进度指示器
            const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
            let node;
            let checkedCount = 0;
            while ((node = walker.nextNode())) {
                const t = node.nodeValue.trim();
                if (!t) continue;
                checkedCount++;
                // 先尝试单节点完整匹配
                let m = t.match(/标注中[（(](\d+)\s*[\/／]\s*(\d+)[）)]/);
                if (!m && t.includes('标注中')) {
                    // "标注中" 与 "(N / M)" 是同父元素下的两个独立文本节点，合并父元素文本再匹配
                    const parentText = (node.parentElement?.textContent || '').replace(/\s+/g, '');
                    m = parentText.match(/标注中[（(](\d+)[\/／](\d+)[）)]/);
                    if (m) dbg('subQId', `通过父元素合并文本匹配: "${node.parentElement?.textContent?.trim()}"`);
                }
                if (m) {
                    dbg('subQId', `匹配"标注中"格式成功: → ${m[1]}_${m[2]}`);
                    return `${m[1]}_${m[2]}`;
                }
            }
            dbgWarn('subQId', `未找到"标注中(...)"文本节点，共扫描 ${checkedCount} 个非空节点`);

            // 降级：原 Q-Part 格式
            const pattern = /Q\d+-Part\s+[A-Z][-–][A-Z]\.\d+/;
            const textSrcDivs = document.querySelectorAll('[class*="textSrc"] > div');
            dbg('subQId', `降级：检查 ${textSrcDivs.length} 个 [textSrc]>div 元素`);
            for (const el of textSrcDivs) {
                const t = el.innerText.trim();
                dbg('subQId', `  textSrc内容: "${t.substring(0, 80)}"`);
                if (pattern.test(t)) {
                    dbg('subQId', `  匹配 Q-Part 格式: ${t}`);
                    return t;
                }
            }
            dbgErr('subQId', '完全失败，返回 null。请检查页面是否有"标注中(N/M)"或 Q-Part 文本');
            return null;
        }

        function getCurrentMarkItemIndex() {
            // 优先从"标注中(N/M)"读取
            const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
            let node;
            while ((node = walker.nextNode())) {
                const t = node.nodeValue.trim();
                if (!t) continue;
                let m = t.match(/标注中[（(](\d+)\s*[\/／]\s*(\d+)[）)]/);
                if (!m && t.includes('标注中')) {
                    const parentText = (node.parentElement?.textContent || '').replace(/\s+/g, '');
                    m = parentText.match(/标注中[（(](\d+)[\/／](\d+)[）)]/);
                }
                if (m) {
                    const idx = parseInt(m[1]) - 1;
                    dbg('markIdx', `从父元素合并文本解析得 index=${idx}`);
                    return idx;
                }
            }
            // 降级：markItem current 类
            const items = document.querySelectorAll('[class*="markItem___"]');
            dbg('markIdx', `降级：找到 ${items.length} 个 markItem___，逐一检查 className`);
            for (let i = 0; i < items.length; i++) {
                dbg('markIdx', `  markItem[${i}] className="${items[i].className}"`);
                if (items[i].className.includes('current___')) {
                    dbg('markIdx', `  命中 current___，index=${i}`);
                    return i;
                }
            }
            dbgWarn('markIdx', '未找到 current markItem，返回 -1');
            return -1;
        }

        function extractMarkingCriteria() {
            for (const el of document.querySelectorAll('[class*="textSrc"]')) {
                const t = el.textContent.trim();
                if (t.startsWith('[[') || t.startsWith('["')) return t;
                if (t.startsWith('[') && t.includes('Award')) return t;
            }
            const m = document.body.innerText.match(/(\[\s*\[.*?\]\s*\])/s);
            return m ? m[1] : null;
        }

        function extractNerText() {
            const container = document.querySelector('[id$="-ner"]');
            if (!container) return null;
            const spans = container.querySelectorAll('.text-label-letter');
            if (!spans.length) return null;
            return Array.from(spans).map(s => s.textContent).join('');
        }

        function getNerContainer() {
            return document.querySelector('[id$="-ner"]');
        }

        function findRemarksInput() {
            // 查找"其他备注"区块下的输入框
            for (const q of document.querySelectorAll('[class*="question___"]')) {
                const title = q.querySelector('[class*="title___"]');
                if (title && /其他备注|备注|remark/i.test(title.textContent)) {
                    const inp = q.querySelector('textarea, [class*="input___"]');
                    dbg('remarks', `找到备注区块，input=${inp ? inp.tagName : 'null'}`);
                    return inp || null;
                }
            }
            dbg('remarks', '未找到"其他备注"区块');
            return null;
        }

        function findScoreInput() {
            const questions = document.querySelectorAll('[class*="question___"]');
            dbg('scoreInput', `找到 ${questions.length} 个 question___ 区块`);
            for (const q of questions) {
                const title = q.querySelector('[class*="title___"]');
                const titleText = title ? title.textContent.trim() : '(无title)';
                dbg('scoreInput', `  question title="${titleText}"`);
                if (title && (title.textContent.includes('总分') || title.textContent.includes('score'))) {
                    const inp = q.querySelector('textarea, [class*="input___"]');
                    dbg('scoreInput', `  命中总分区块，input=${inp ? inp.tagName + '[class=' + inp.className.substring(0, 50) + ']' : 'null'}`);
                    if (inp) return inp;
                }
            }
            const fallback = document.querySelector('[class*="input___"]');
            dbgWarn('scoreInput', `未找到总分区块，降级 fallback input=${fallback ? fallback.tagName : 'null'}`);
            return fallback;
        }

        // ==========================================
        // 6. React 受控组件值设置
        // ==========================================
        function setReactInputValue(element, value) {
            if (!element) return false;
            const strVal = String(value);
            try {
                const proto = element.tagName === 'TEXTAREA'
                    ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
                const nativeSetter = Object.getOwnPropertyDescriptor(proto, 'value').set;

                // 先清空
                nativeSetter.call(element, '');
                element.dispatchEvent(new InputEvent('input', { bubbles: true, cancelable: true }));

                // 再写入新值（使用 InputEvent 而非普通 Event，React 能正确感知）
                nativeSetter.call(element, strVal);
                element.dispatchEvent(new InputEvent('input', {
                    bubbles: true, cancelable: true,
                    data: strVal, inputType: 'insertText'
                }));
                element.dispatchEvent(new Event('change', { bubbles: true }));
                return element.value === strVal;
            } catch (e) {
                log('setReactInputValue 失败: ' + e.message);
                return false;
            }
        }

        // ==========================================
        // 7. 查找弹窗中的输入框和关闭按钮
        // ==========================================

        /**
         * 等待双击后弹出的标注窗口出现，最多等待 timeoutMs
         * 返回 { input, closeBtn } 或 null
         */
        async function waitForAnnotationPopup(timeoutMs = 1500) {
            const start = Date.now();
            while (Date.now() - start < timeoutMs) {
                // 找到包含 input 且附近有"关闭"按钮的浮动容器
                const result = findPopupElements();
                if (result) return result;
                await sleep(60);
            }
            return null;
        }

        /**
         * 检测 iLabel 双击后出现的"快捷键提示"浮层（含"Ctrl+C"/"Enter"文字但无 score input）
         * 如果存在，需要按 Enter 或点击"Enter"才能进入真正的标注弹窗
         */
        function findShortcutHintPopup() {
            const selectors = [
                '[class*="guide"]', '[class*="hint"]', '[class*="shortcut"]',
                '[class*="tip"]', '[class*="popover"]', '[class*="tooltip"]',
                '[class*="contextmenu"]', '[class*="context-menu"]',
            ];
            for (const sel of selectors) {
                for (const el of document.querySelectorAll(sel)) {
                    if (!el.offsetParent) continue;
                    const text = el.textContent;
                    // 含 Ctrl 字样且没有可输入的 input（排除真正的标注弹窗）
                    if (/ctrl|⌘|⌥/i.test(text) && !el.querySelector('input:not([type="hidden"])')) {
                        return el;
                    }
                }
            }
            return null;
        }

        /** input 或 textarea 均视为可填写的标注输入框 */
        function findInputOrTextarea(container) {
            return container.querySelector('input:not([type="hidden"]), textarea');
        }

        function findPopupElements() {
            // 策略1: ant-design Modal（silkQ 标注弹窗）— 优先精确匹配
            for (const sel of ['.ant-modal-content', '.ant-popover-inner', '.ant-tooltip-inner']) {
                const el = document.querySelector(sel);
                if (!el) { dbg('popup', `策略1: ${sel} 不存在`); continue; }
                if (!el.offsetParent) { dbg('popup', `策略1: ${sel} 存在但不可见`); continue; }
                const inp = findInputOrTextarea(el);
                if (!inp) { dbg('popup', `策略1: ${sel} 可见但无 input/textarea`); continue; }
                const closeBtn = findCloseButton(el);
                if (!closeBtn) { dbg('popup', `策略1: ${sel} 有 input/textarea 但无关闭按钮，文本="${el.textContent.substring(0,60)}"`); continue; }
                dbg('popup', `策略1 命中: ${sel}，元素=${inp.tagName}`);
                return { input: inp, closeBtn, container: el };
            }

            // 策略2: 所有含 popover/popup/dialog/modal 关键词的浮层
            const floatSelectors = [
                '[class*="popover"]', '[class*="popup"]', '[class*="dialog"]',
                '[class*="modal"]', '[class*="overlay"]', '[class*="labelInput"]',
                '[class*="annotationPopup"]', '[class*="tag-input"]',
            ];
            for (const sel of floatSelectors) {
                const els = document.querySelectorAll(sel);
                if (els.length) dbg('popup', `策略2: ${sel} 找到 ${els.length} 个元素`);
                for (const container of els) {
                    if (!container.offsetParent) continue;
                    const inp = findInputOrTextarea(container);
                    if (!inp) continue;
                    const closeBtn = findCloseButton(container);
                    if (!closeBtn) {
                        dbg('popup', `策略2: ${sel} 有 ${inp.tagName} 但无关闭按钮，class="${container.className.substring(0,60)}"`);
                        continue;
                    }
                    dbg('popup', `策略2 命中: ${sel}，${inp.tagName}.value="${inp.value.substring(0,30)}"`);
                    return { input: inp, closeBtn, container };
                }
            }

            // 策略3: 通用回退——可见 input/textarea 向上找最近的小尺寸容器（含关闭按钮）
            const visibleFields = Array.from(
                document.querySelectorAll('input:not([type="hidden"]), textarea')
            ).filter(i => i.offsetParent);
            dbg('popup', `策略3: 可见 input/textarea 共 ${visibleFields.length} 个`);
            for (const inp of visibleFields) {
                let ancestor = inp.parentElement;
                for (let d = 0; d < 10 && ancestor; d++) {
                    const closeBtn = findCloseButton(ancestor);
                    if (closeBtn) {
                        const rect = ancestor.getBoundingClientRect();
                        dbg('popup', `策略3: 关闭按钮在祖先[${d}]，容器 ${rect.width.toFixed(0)}×${rect.height.toFixed(0)}`);
                        if (rect.width < 900 && rect.height < 700) {
                            dbg('popup', `策略3 命中，${inp.tagName}`);
                            return { input: inp, closeBtn, container: ancestor };
                        }
                    }
                    ancestor = ancestor.parentElement;
                }
            }

            dbgErr('popup', '所有策略均失败。可见 input/textarea:',
                visibleFields.map(i => `${i.tagName} class=${i.className.substring(0,40)}`));
            return null;
        }

        function findCloseButton(container) {
            if (!container) return null;
            return Array.from(container.querySelectorAll('button, [class*="btn"], [class*="close"]'))
                .find(el => {
                    // 去掉所有内部空白再匹配（处理"关 闭"、"确 定"等中间含空格的情况）
                    const text = el.textContent.replace(/\s+/g, '');
                    return /关闭|close|确定|confirm|^ok$/i.test(text);
                });
        }

        // ==========================================
        // 8. 核心：NER 文本定位 + 双击标注
        // ==========================================

        /**
         * 在 NER 容器中找到 textToFind 对应的 span 索引范围
         * 三级降级匹配：精确 → 折叠空白 → 前N字符截断
         */
        function findTextSpanRange(nerContainer, textToFind) {
            if (!textToFind || !nerContainer) return null;

            const spans = Array.from(nerContainer.querySelectorAll('.text-label-letter'));
            if (!spans.length) { dbgErr('spanRange', '未找到任何 .text-label-letter span'); return null; }

            const fullText = spans.map(s => s.textContent).join('');
            const searchPreview = textToFind.substring(0, 60);

            /**
             * 若 textToFind 是不完整公式（以定界符开头但未以对应定界符结尾），
             * 将 end 自动延伸到 fullText 中下一个闭合定界符的位置。
             * 支持: $$...$$  $...$  \[...\]  \(...\)
             */
            function extendEnd(end) {
                const t = textToFind.trimEnd();
                let close = null;
                if      (t.startsWith('$$') && !t.endsWith('$$'))         close = '$$';
                else if (!t.startsWith('$$') && t.startsWith('$') && !t.endsWith('$'))
                                                                           close = '$';
                else if (t.startsWith('\\[') && !t.endsWith('\\]'))       close = '\\]';
                else if (t.startsWith('\\(') && !t.endsWith('\\)'))       close = '\\)';
                if (!close) return end;
                const closeIdx = fullText.indexOf(close, end + 1);
                if (closeIdx < 0) {
                    dbgWarn('spanRange', `不完整公式但未找到闭合 "${close}"，end 保持 ${end}`);
                    return end;
                }
                const newEnd = Math.min(closeIdx + close.length - 1, spans.length - 1);
                dbg('spanRange', `不完整公式自动延伸(${close}): end ${end}→${newEnd}`);
                return newEnd;
            }

            // 精确匹配
            let idx = fullText.indexOf(textToFind);
            if (idx >= 0) {
                const end = extendEnd(idx + textToFind.length - 1);
                dbg('spanRange', `精确匹配 ✓ "${searchPreview}" → [${idx}-${end}]`);
                return { start: idx, end, spans };
            }

            // 折叠空白匹配
            const normFull = fullText.replace(/\s+/g, ' ');
            const normSearch = textToFind.replace(/\s+/g, ' ').trim();
            const normIdx = normFull.indexOf(normSearch);
            if (normIdx >= 0) {
                // 将 normFull 中的位置映射回 fullText 中的位置
                let origStart = -1, origEnd = -1, normPos = 0;
                for (let i = 0; i < fullText.length; i++) {
                    const c = fullText[i];
                    if (normPos === normIdx && origStart < 0) origStart = i;
                    if (normPos === normIdx + normSearch.length - 1) { origEnd = i; break; }
                    if (/\s/.test(c)) {
                        if (normPos < normFull.length && normFull[normPos] === ' ') {
                            normPos++;
                            while (i + 1 < fullText.length && /\s/.test(fullText[i + 1])) i++;
                        }
                    } else {
                        normPos++;
                    }
                }
                if (origStart >= 0 && origEnd >= 0) {
                    origEnd = extendEnd(origEnd);
                    dbg('spanRange', `折叠空白匹配 ✓ "${searchPreview}" → [${origStart}-${origEnd}]`);
                    return { start: origStart, end: origEnd, spans };
                }
            }
            dbg('spanRange', `精确/折叠空白均失败，尝试截断: "${searchPreview}"`);

            // 前 N 字符截断匹配（仅用于定位起始位置，end 按完整文本长度估算）
            const shortSearch = normSearch.substring(0, Math.min(30, normSearch.length));
            if (shortSearch.length >= 5) {
                const shortIdx = normFull.indexOf(shortSearch);
                if (shortIdx >= 0) {
                    let origStart = -1, normPos = 0;
                    for (let i = 0; i < fullText.length; i++) {
                        if (normPos === shortIdx) { origStart = i; break; }
                        const c = fullText[i];
                        if (/\s/.test(c)) {
                            if (normPos < normFull.length && normFull[normPos] === ' ') {
                                normPos++;
                                while (i + 1 < fullText.length && /\s/.test(fullText[i + 1])) i++;
                            }
                        } else {
                            normPos++;
                        }
                    }
                    // Bug fix: 用完整 textToFind 的长度估算 end，而非 shortSearch 的长度
                    let origEnd = Math.min(origStart + textToFind.length - 1, spans.length - 1);
                    origEnd = extendEnd(origEnd);
                    if (origStart >= 0) {
                        dbg('spanRange', `前30字截断匹配 ✓ "${shortSearch}" → [${origStart}-${origEnd}]`);
                        return { start: origStart, end: origEnd, spans };
                    }
                }
                dbg('spanRange', `前30字截断失败: "${shortSearch}"`);
            }

            // 首尾锚点匹配：处理 LLM 省略了公式中间内容的情况
            {
                const anchorLen = Math.min(22, Math.floor(normSearch.length / 3));
                if (anchorLen >= 6) {
                    const headAnchor = normSearch.substring(0, anchorLen);
                    const tailAnchor = normSearch.substring(normSearch.length - anchorLen);
                    const headOrigIdx = fullText.indexOf(headAnchor);
                    if (headOrigIdx >= 0) {
                        const tailOrigIdx = fullText.indexOf(tailAnchor, headOrigIdx + anchorLen);
                        if (tailOrigIdx >= 0) {
                            const end = extendEnd(tailOrigIdx + tailAnchor.length - 1);
                            dbg('spanRange', `首尾锚点匹配 ✓ head="${headAnchor}" / tail="${tailAnchor}" → [${headOrigIdx}-${end}]`);
                            return { start: headOrigIdx, end, spans };
                        }
                        dbg('spanRange', `首尾锚点：head="${headAnchor}" ✓，tail="${tailAnchor}" ✗`);
                    } else {
                        dbg('spanRange', `首尾锚点：head="${headAnchor}" ✗`);
                    }
                }
            }

            // 定界符格式替换匹配：
            // LLM 常把 \[...\] 写成 $$...$$（或反之），本策略提取数学核心内容
            // 在 NER 中无定界符限制地查找，再向外扩展到实际定界符。
            {
                // 从 textToFind 剥离外层数学定界符，提取纯核心内容
                let mathCore = null;
                let cm;
                const tf = textToFind.trim();
                if      ((cm = tf.match(/^\$\$([\s\S]+)\$\$$/)))          mathCore = cm[1].trim();
                else if ((cm = tf.match(/^\\\[([\s\S]+)\\\]$/)))           mathCore = cm[1].trim();
                else if ((cm = tf.match(/^\\\(([\s\S]+)\\\)$/)))           mathCore = cm[1].trim();
                else if ((cm = tf.match(/^\$(?!\$)([\s\S]+[^$])\$(?!\$)/)))mathCore = cm[1].trim();

                if (mathCore && mathCore.length >= 8) {
                    const normCore = mathCore.replace(/\s+/g, ' ');
                    const coreNormIdx = normFull.indexOf(normCore);
                    if (coreNormIdx >= 0) {
                        // 将 normFull 位置映射回 fullText
                        let cs = -1, ce = -1, np = 0;
                        for (let i = 0; i < fullText.length; i++) {
                            const c = fullText[i];
                            if (np === coreNormIdx && cs < 0) cs = i;
                            if (np === coreNormIdx + normCore.length - 1) { ce = i; break; }
                            if (/\s/.test(c)) {
                                if (np < normFull.length && normFull[np] === ' ') {
                                    np++;
                                    while (i + 1 < fullText.length && /\s/.test(fullText[i + 1])) i++;
                                }
                            } else { np++; }
                        }

                        if (cs >= 0 && ce >= 0) {
                            // 向前延伸：吃掉核心内容前面的开始定界符
                            for (const d of ['$$', '\\[', '\\(', '$']) {
                                if (cs >= d.length && fullText.substring(cs - d.length, cs) === d) {
                                    cs -= d.length; break;
                                }
                            }
                            // 向后延伸：先跳过标点（., ;），再吃掉闭合定界符
                            let ep = ce + 1;
                            while (ep < fullText.length && /[.,;!?]/.test(fullText[ep])) ep++;
                            for (const d of ['$$', '\\]', '\\)', '$']) {
                                if (fullText.substring(ep, ep + d.length) === d) {
                                    ce = ep + d.length - 1; break;
                                }
                            }
                            dbg('spanRange', `定界符替换匹配 ✓ core="${normCore.substring(0, 40)}" → [${cs}-${ce}]`);
                            return { start: cs, end: ce, spans };
                        }
                    }
                    dbg('spanRange', `定界符替换匹配失败: core="${normCore.substring(0, 40)}"`);
                }
            }

            dbgErr('spanRange', `所有匹配均失败 → "${textToFind.substring(0, 80)}"`);
            return null;
        }

        /**
         * 对指定 span 范围执行：选中 → 双击 → 等弹窗 → 输入分值 → 关闭
         * @param {Element} nerContainer
         * @param {number} startIdx
         * @param {number} endIdx
         * @param {number|string} scoreValue  要输入的分值
         * @returns {boolean}
         */
        async function annotateSpanRange(nerContainer, startIdx, endIdx, scoreValue) {
            const spans = nerContainer.querySelectorAll('.text-label-letter');
            if (startIdx >= spans.length || endIdx >= spans.length || startIdx > endIdx) {
                log(`span 索引越界: ${startIdx}-${endIdx}，总数 ${spans.length}`);
                return false;
            }

            const startSpan = spans[startIdx];
            const endSpan   = spans[endIdx];
            const midSpan   = spans[Math.floor((startIdx + endIdx) / 2)];

            // Step 1: 创建 DOM Range 选区
            try {
                const range = document.createRange();
                range.setStart(startSpan.firstChild || startSpan, 0);
                const endNode = endSpan.firstChild || endSpan;
                range.setEnd(endNode, endNode.nodeValue ? endNode.nodeValue.length : 1);
                const sel = window.getSelection();
                sel.removeAllRanges();
                sel.addRange(range);
            } catch (_) {}

            // Step 2: mousedown → mouseup 触发选择（React 层）
            const startRect = startSpan.getBoundingClientRect();
            const endRect   = endSpan.getBoundingClientRect();

            startSpan.dispatchEvent(new MouseEvent('mousedown', {
                bubbles: true, cancelable: true, button: 0, buttons: 1,
                clientX: startRect.left + 2, clientY: startRect.top + 2
            }));
            await sleep(60);
            endSpan.dispatchEvent(new MouseEvent('mouseup', {
                bubbles: true, cancelable: true, button: 0, buttons: 0,
                clientX: endRect.right - 2, clientY: endRect.bottom - 2
            }));
            await sleep(120);

            // Step 3: dblclick 触发 iLabel 弹窗（可能先出现快捷键提示，再出现标注输入框）
            const midRect = midSpan.getBoundingClientRect();
            midSpan.dispatchEvent(new MouseEvent('dblclick', {
                bubbles: true, cancelable: true, button: 0,
                clientX: midRect.left + 2, clientY: midRect.top + 2
            }));
            await sleep(350);

            // Step 4: 若出现"Ctrl+C / Enter"快捷键提示框，自动按 Enter 进入标注弹窗
            const hint = findShortcutHintPopup();
            if (hint) {
                log('检测到快捷键提示框，按 Enter 继续...');
                // 优先点击提示框内的 Enter 相关按钮
                const enterBtn = Array.from(hint.querySelectorAll('button, [class*="btn"], span[role="button"]'))
                    .find(el => /^enter$|^确认$|^确定$/i.test(el.textContent.trim()));
                if (enterBtn) {
                    enterBtn.click();
                } else {
                    hint.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', code: 'Enter', keyCode: 13, bubbles: true }));
                    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', code: 'Enter', keyCode: 13, bubbles: true }));
                }
                await sleep(350);
            }

            // Step 5: 等待真正的标注弹窗（含 score input + 关闭按钮）
            dbg('annotate', `等待弹窗（最多2000ms）...`);
            const popup = await waitForAnnotationPopup(2000);
            if (!popup) {
                dbgErr('annotate', '未检测到标注弹窗，跳过此条。请确认双击是否触发了 iLabel 的标注浮层');
                log('⚠ 未检测到标注弹窗，跳过此条');
                return false;
            }
            dbg('annotate', `弹窗已找到: input.type="${popup.input.type}" input.value="${popup.input.value}" closeBtn.text="${popup.closeBtn.textContent.trim()}"`);

            // Step 6: 向输入框写入分值
            popup.input.focus();
            await sleep(80);
            dbg('annotate', `准备写入分值 "${scoreValue}"，当前 input.value="${popup.input.value}"`);

            // 方法A：execCommand（模拟真实用户输入，React 各版本均可感知）
            popup.input.select?.();
            document.execCommand('selectAll', false, null);
            const inserted = document.execCommand('insertText', false, String(scoreValue));
            dbg('annotate', `execCommand insertText 返回 ${inserted}，写入后 input.value="${popup.input.value}"`);

            if (!inserted || popup.input.value !== String(scoreValue)) {
                dbgWarn('annotate', `execCommand 未生效（期望="${scoreValue}" 实际="${popup.input.value}"），改用 setReactInputValue`);
                // 方法B：React native setter + InputEvent
                setReactInputValue(popup.input, String(scoreValue));
                dbg('annotate', `setReactInputValue 后 input.value="${popup.input.value}"`);
            }
            await sleep(100);

            // 确认输入（部分弹窗需要 Enter 才能提交）
            popup.input.dispatchEvent(new KeyboardEvent('keydown', {
                key: 'Enter', code: 'Enter', keyCode: 13, bubbles: true
            }));
            await sleep(60);

            // Step 6.5: 最终验证输入值是否写入成功，否则中止（避免以空值或错误值提交）
            const finalInputValue = popup.input.value;
            if (finalInputValue !== String(scoreValue)) {
                dbgErr('annotate', `分值写入最终失败: 期望="${scoreValue}"，实际="${finalInputValue}"，跳过关闭`);
                log(`✗ 分值写入失败（期望${scoreValue}，实际${finalInputValue}）`);
                return false;
            }

            // Step 7: 点击"关闭"按钮，等待弹窗动画完全结束
            dbg('annotate', `点击关闭按钮: "${popup.closeBtn.textContent.trim()}"`);
            popup.closeBtn.click();
            await sleep(1500);

            // Step 8: 验证弹窗已关闭（元素从 DOM 移除或宽高归零均视为关闭成功）
            const popupClosed = !document.contains(popup.container) ||
                popup.container.getBoundingClientRect().width === 0;
            if (!popupClosed) {
                dbgWarn('annotate', '弹窗关闭后仍然可见，标注可能未保存');
                return false;
            }

            dbg('annotate', `✓ 标注完成，scoreValue=${scoreValue}`);
            log(`✓ 已标注分值 ${scoreValue}`);
            return true;
        }

        // ==========================================
        // 9. 标注流程编排
        // ==========================================

        async function annotateOneCriterion(result, criterionIndex) {
            if (!result.satisfied || !result.matched_text) return false;

            const nerContainer = getNerContainer();
            if (!nerContainer) { log('未找到NER容器'); return false; }

            const matchedText = result.matched_text;
            log(`[${criterionIndex + 1}] 查找文本: "${matchedText.substring(0, 45)}..."`);

            const spanRange = findTextSpanRange(nerContainer, matchedText);
            if (!spanRange) {
                log(`⚠ [${criterionIndex + 1}] 未找到匹配文本`);
                return false;
            }

            const scoreValue = result.points_awarded;
            const MAX_RETRIES = 2;
            for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
                if (attempt > 1) {
                    log(`⚠ [${criterionIndex + 1}] 第${attempt}次重试标注...`);
                    await sleep(1200);
                }
                const ok = await annotateSpanRange(nerContainer, spanRange.start, spanRange.end, scoreValue);
                if (ok) {
                    log(`✓ [${criterionIndex + 1}] 标注完成，分值=${scoreValue} (span ${spanRange.start}-${spanRange.end})`);
                    await sleep(3500); // 关闭按钮后 1500ms + 此处 3500ms = 两次标注间共 5000ms
                    return true;
                }
            }
            log(`✗ [${criterionIndex + 1}] 标注失败（已重试${MAX_RETRIES}次）`);
            return false;
        }

        async function applyAllAnnotations(data, subQId) {
            const results = data.results || [];
            showResult(data, subQId);

            // ── 打印所有结果的原始数据，方便核查哪条被跳过 ──
            dbg('applyAll', `共 ${results.length} 条结果，逐条列出：`);
            results.forEach((r, i) => {
                dbg('applyAll',
                    `  [${i + 1}] satisfied=${r.satisfied}  points_awarded=${r.points_awarded}` +
                    `  matched_text="${(r.matched_text || '').substring(0, 60)}"` +
                    `  criterion="${(r.criterion || '').substring(0, 60)}"`
                );
            });

            let annotated = 0;
            const failedAnnotations = [];
            for (let i = 0; i < results.length; i++) {
                const r = results[i];
                const shouldAnnotate = r.satisfied && r.matched_text;
                dbg('applyAll', `[${i + 1}] shouldAnnotate=${shouldAnnotate}` +
                    (shouldAnnotate ? '' : `（satisfied=${r.satisfied}, matched_text=${!!r.matched_text}）`));
                if (shouldAnnotate) {
                    const ok = await annotateOneCriterion(r, i);
                    if (ok) {
                        annotated++;
                    } else {
                        failedAnnotations.push({
                            criterionIndex: i,
                            criterion: r.criterion,
                            matchedText: r.matched_text
                        });
                    }
                } else {
                    log(`跳过 [${i + 1}]（未满足或无匹配文本）`);
                }
            }

            // 填写总分（格式：0.2+0.3+0=0.5）
            dbg('totalScore', `开始填写总分，data.score_string="${data.score_string}" data.total_score=${data.total_score}`);
            const scoreInput = findScoreInput();
            if (scoreInput) {
                const val = data.score_string || String(data.total_score || 0);
                dbg('totalScore', `目标输入框 tagName=${scoreInput.tagName} class="${scoreInput.className.substring(0,60)}" 当前值="${scoreInput.value}"`);
                const ok = setReactInputValue(scoreInput, val);
                dbg('totalScore', `setReactInputValue 返回 ${ok}，写入后 scoreInput.value="${scoreInput.value}"`);
                if (!ok) dbgErr('totalScore', `总分填写失败！期望="${val}" 实际="${scoreInput.value}"`);
                log(`总分${ok ? '已填写' : '填写失败'}: ${val}`);
            } else {
                dbgErr('totalScore', '未找到总分输入框，请检查 findScoreInput 逻辑与页面 class 名称');
                log('⚠ 未找到总分输入框');
            }

            // 填写"其他备注"（仅当模型返回了 note 字段时）
            if (data.note) {
                const remarksInput = findRemarksInput();
                if (remarksInput) {
                    const ok = setReactInputValue(remarksInput, data.note);
                    dbg('remarks', `备注填写${ok ? '成功' : '失败'}: "${data.note}"`);
                    log(`备注${ok ? '已填写' : '填写失败'}: ${data.note}`);
                } else {
                    dbgWarn('remarks', `有备注但未找到输入框: "${data.note}"`);
                }
            }

            const failMsg = failedAnnotations.length > 0 ? `，失败 ${failedAnnotations.length} 条` : '';
            log(`完成！标注 ${annotated}/${results.length} 条${failMsg}，总分: ${data.score_string || '0'}`);
            return { annotated, total: results.length, failed: failedAnnotations };
        }

        // ==========================================
        // 10. API 调用
        // ==========================================

        function callPhysicsAPI(taskId, marking, modelAnswer) {
            return new Promise((resolve, reject) => {
                GM_xmlhttpRequest({
                    method: 'POST',
                    url: 'http://localhost:8001/api/physics_task',
                    headers: { 'Content-Type': 'application/json' },
                    data: JSON.stringify({ task_id: taskId, marking: marking || '[]', model_answer: modelAnswer || '' }),
                    onload(resp) {
                        try { resolve(JSON.parse(resp.responseText)); }
                        catch (e) { reject(new Error('API响应解析失败: ' + e.message)); }
                    },
                    onerror() { reject(new Error('API请求失败，请确认本地服务已启动')); }
                });
            });
        }

        // 无限轮询（后端已配置无限重试，前端同步等待直到后端返回 finished 或 error）
        function pollPhysicsResult(taskId) {
            return new Promise((resolve, reject) => {
                let tries = 0;
                const timer = setInterval(() => {
                    tries++;
                    GM_xmlhttpRequest({
                        method: 'GET',
                        url: `http://localhost:8001/api/physics_result/${encodeURIComponent(taskId)}`,
                        onload(resp) {
                            try {
                                const res = JSON.parse(resp.responseText);
                                if (res.status === 'finished') { clearInterval(timer); resolve(res.data); }
                                else if (res.status === 'error') { clearInterval(timer); reject(new Error(res.error || '评分出错')); }
                                // pending / 其他状态：继续等待，不做任何处理
                            } catch (_) {}
                        }
                    });
                    // 每 5 分钟在悬浮窗提示一次等待进度
                    if (tries % 120 === 0) log(`仍在等待评分结果，已等待约 ${tries * 2.5 / 60 | 0} 分钟...`);
                }, 2500);
            });
        }

        /**
         * 防碰撞跳过流程：点击"跳过"→填写理由→点击弹窗内的确认跳过按钮
         * 与 ilabel.js 的 executeSkipSequence 逻辑相同
         */
        async function executePhysSkip(collider) {
            log(`⚡ 冲突（被 ${collider} 占用），执行自动跳过...`);
            debugPanel.style.borderColor = '#f5222d';
            setTimeout(() => { debugPanel.style.borderColor = '#1a3a6a'; }, 600);

            const skipBtn = Array.from(document.querySelectorAll('button'))
                .find(btn => btn.innerText.replace(/\s+/g, '') === '跳过');
            if (!skipBtn) { log('⚠ 未找到跳过按钮'); return; }

            skipBtn.click();
            await sleep(700);

            // 填写理由（textarea id="reason"）
            const textarea = document.getElementById('reason');
            if (!textarea) { log('⚠ 未找到跳过理由输入框'); return; }

            const nativeSetter = Object.getOwnPropertyDescriptor(
                unsafeWindow.HTMLTextAreaElement.prototype, 'value'
            ).set;
            nativeSetter.call(textarea, '双线程自动防碰撞跳过');
            textarea.dispatchEvent(new Event('input', { bubbles: true }));
            await sleep(350);

            // 点击弹窗内的"跳过"确认按钮
            const confirmBtn = Array.from(
                document.querySelectorAll('.ant-modal-footer button, div[style*="flex-end"] button')
            ).find(btn => btn.innerText.replace(/\s+/g, '') === '跳过');

            if (!confirmBtn) { log('⚠ 未找到跳过确认按钮'); return; }
            confirmBtn.click();
            log('✓ 自动跳过完成');
        }

        /** 查找"下一题"按钮 */
        function findNextButton() {
            return Array.from(document.querySelectorAll('button'))
                .find(btn => /下一题|下一个|next/i.test(btn.textContent.replace(/\s+/g, '')));
        }

        /** 查找"提交"/"提交任务"按钮 */
        function findSubmitButton() {
            return Array.from(document.querySelectorAll('button'))
                .find(btn => /^提交$|^提交任务$|^submit$/i.test(btn.textContent.replace(/\s+/g, '')));
        }

        /**
         * 自动推进：点击"下一题"或"提交"，点击后等待检测是否跳转成功，
         * 若未成功则重试，最多 MAX_ADVANCE_ATTEMPTS 次。
         */
        async function autoAdvance(taskId, subQId) {
            const MAX_ATTEMPTS = 6;
            const WAIT_MS = 5000; // 点击后等待检测的时间
            const subQParts = subQId.match(/^(\d+)_(\d+)$/);
            const isLast = subQParts && subQParts[1] === subQParts[2];

            for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
                if (isLast) {
                    const btn = findSubmitButton();
                    if (!btn) { log('⚠ 未找到提交按钮，跳过'); return; }
                    log(`末题，自动提交...${attempt > 1 ? `（第${attempt}次）` : ''}`);
                    btn.click();
                } else {
                    const btn = findNextButton();
                    if (!btn) { log('⚠ 未找到下一题按钮，跳过'); return; }
                    log(`自动点击下一题...${attempt > 1 ? `（第${attempt}次）` : ''}`);
                    btn.click();
                }

                await sleep(WAIT_MS);

                const newTaskId  = getTaskUUID();
                const newSubQId  = getSubQuestionId();
                // 判断是否已跳转：
                //   - UUID 变了 → 一定到新任务了（含"1_1"提交后仍是"1_1"的情况）
                //   - subQId 消失或变了 → 也算成功
                const navigated = isLast
                    ? (newTaskId !== taskId || !newSubQId || newSubQId !== subQId)
                    : (newSubQId && newSubQId !== subQId);

                if (navigated) {
                    log(`✓ 推进成功 → ${newSubQId || '(已提交)'}`);
                    return;
                }

                if (attempt < MAX_ATTEMPTS) {
                    log(`⚠ 未检测到跳转，重试（${attempt}/${MAX_ATTEMPTS}）...`);
                    await sleep(1500);
                } else {
                    log('⚠ 自动推进多次失败，请手动操作');
                }
            }
        }

        // ==========================================
        // 11. 主处理流程
        // ==========================================

        async function processSubQuestion() {
            if (isProcessing) return;

            const taskId = getTaskUUID();
            const subQId = getSubQuestionId();
            if (!taskId || !subQId) { updateStatus('等待页面加载...'); return; }

            const processKey = `${taskId}__${subQId}`;
            if (processKey === lastProcessedKey) return;

            // 防碰撞：检查其他标签页是否已在处理同一子题
            const collider = checkPhysCollision(processKey);
            if (collider) {
                dbgWarn('collision', `processKey="${processKey}" 被 ${collider} 占用`);
                await executePhysSkip(collider);
                // 跳过后等待新任务页面加载完毕，再重新检测并开始标注
                log('等待新任务加载...');
                await sleep(3000);
                processSubQuestion();
                return;
            }
            registerPhysTask(processKey);

            isProcessing = true;
            updateStatus(`检测到子题: ${subQId}`);

            try {
                await sleep(1200); // 等待 NER 区域渲染

                const marking = extractMarkingCriteria();
                const nerText = extractNerText();

                if (!marking) { log('⚠ 未找到marking标准，等待页面加载'); isProcessing = false; return; }
                if (!nerText || nerText.length < 10) { log('⚠ NER文本为空，稍后重试'); isProcessing = false; return; }

                log(`marking: ${marking.substring(0, 60)}...`);
                log(`NER文本: ${nerText.length} 字符`);

                const apiTaskId = taskId + '_' + subQId.replace(/[\s\/]/g, '_');
                showPending(subQId);

                await callPhysicsAPI(apiTaskId, marking, nerText);
                log('评分请求已提交，轮询中...');

                const result = await pollPhysicsResult(apiTaskId);
                log(`收到结果，总分: ${result.total_score}`);

                lastProcessedKey = processKey; // 标记为已处理
                const annotationResult = await applyAllAnnotations(result, subQId);

                // 返修模式：若有标注失败的条目，写入跨标签页共享的返修队列
                if (getRevisionMode() && annotationResult.failed.length > 0) {
                    const queueItem = {
                        processKey,
                        taskId,
                        subQId,
                        failedCriteria: annotationResult.failed,
                        timestamp: new Date().toISOString()
                    };
                    const _rq = getRevisionQueue();
                    const existingIdx = _rq.findIndex(q => q.processKey === processKey);
                    if (existingIdx >= 0) {
                        _rq[existingIdx] = queueItem;
                    } else {
                        _rq.push(queueItem);
                    }
                    setRevisionQueue(_rq);
                    log(`⚠ 返修模式: ${annotationResult.failed.length} 条标注失败 → 已加入返修队列 (共${_rq.length}项)`);
                    updateStatus('返修队列更新');
                }

                // 自动推进：末题点提交，否则点下一题（含重试）
                await sleep(1500);
                await autoAdvance(taskId, subQId);

            } catch (e) {
                log('处理出错: ' + e.message);
                console.error('[物理划词]', e);
            } finally {
                releasePhysTask();
                isProcessing = false;
            }
        }

        // ==========================================
        // 12. MutationObserver + 轮询兜底
        // ==========================================

        function setupSubQObserver() {
            if (subQObserver) subQObserver.disconnect();

            subQObserver = new MutationObserver((mutations) => {
                let changed = false;
                for (const m of mutations) {
                    if (m.type === 'attributes' && m.attributeName === 'class') {
                        if (m.target.className?.includes('markItem') && m.target.className?.includes('current___')) {
                            changed = true; break;
                        }
                    }
                    if (m.type === 'childList') {
                        for (const node of m.addedNodes) {
                            if (node.nodeType === 1 &&
                                ((node.id && node.id.endsWith('-ner')) ||
                                (node.className && typeof node.className === 'string' && node.className.includes('text-label-letter')))) {
                                changed = true; break;
                            }
                        }
                        const el = m.target.parentElement;
                        if (el?.className?.includes?.('textSrc')) changed = true;
                    }
                    if (changed) break;
                }
                if (changed && !isProcessing) {
                    clearTimeout(subQObserver._debounce);
                    subQObserver._debounce = setTimeout(processSubQuestion, 800);
                }
            });

            subQObserver.observe(document.body, {
                subtree: true, attributes: true, attributeFilter: ['class'],
                childList: true, characterData: false
            });
            log('MutationObserver 已启动');
        }

        function startMainLoop() {
            let lastKey = null;
            setInterval(() => {
                if (isProcessing) return;
                const taskId = getTaskUUID();
                const subQ   = getSubQuestionId();
                if (!taskId || !subQ) return;
                const key = `${taskId}__${subQ}`;
                if (key === lastKey) return;
                lastKey = key;
                setTimeout(() => { if (!isProcessing) processSubQuestion(); }, 1500);
            }, 1000);
        }

        // ==========================================
        // 13. 启动
        // ==========================================
        setTimeout(() => {
            setupSubQObserver();
            startMainLoop();
            updateStatus('就绪，等待子题切换...');
            setTimeout(processSubQuestion, 2000);
        }, 1500);

        window.addEventListener('beforeunload', () => {
            subQObserver?.disconnect();
            releasePhysTask();
        });

        // 监听其他标签页对返修状态的修改，及时刷新本标签页的悬浮窗
        try {
            (typeof unsafeWindow !== 'undefined' ? unsafeWindow : window)
                .addEventListener('storage', (e) => {
                    if (e.key === PHYS_REVISION_MODE_KEY || e.key === PHYS_REVISION_QUEUE_KEY) {
                        updateStatus('跨标签页同步');
                    }
                });
        } catch (_) {}

    })();
