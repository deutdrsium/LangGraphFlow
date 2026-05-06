// ==UserScript==
// @name         iLabel Physics Score Checker
// @namespace    http://tampermonkey.net/
// @version      1.0
// @description  检查物理题：① 总分栏算式正确性  ② 划词标注总分与总分栏是否一致
// @author       Assistant
// @match        *://ilabel.alipay.com/*
// @match        file:///*
// @grant        none
// ==/UserScript==

(function () {
    'use strict';

    // ── 子题号检测 ────────────────────────────────────────────────────────
    function getSubQuestionId() {
        const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
        let node;
        while ((node = walker.nextNode())) {
            const t = node.nodeValue.trim();
            if (!t) continue;
            let m = t.match(/(?:标注中|修改中)[（(](\d+)\s*[\/／]\s*(\d+)[）)]/);
            if (!m && (t.includes('标注中') || t.includes('修改中'))) {
                const pt = (node.parentElement?.textContent || '').replace(/\s+/g, '');
                m = pt.match(/(?:标注中|修改中)[（(](\d+)[\/／](\d+)[）)]/);
            }
            if (m) return `${m[1]}_${m[2]}`;
        }
        return null;
    }

    // ── 从"标注结果"区域提取各 NER 标注的分值 ────────────────────────────
    // iLabel 的 ant-descriptions 结构：
    //   ant-descriptions-title  → "标注结果:"  （区块标题）
    //   ant-descriptions-item-label   → 分值（如 "0.1"）  每条标注一个
    //   ant-descriptions-item-content → 对应的标注文本 tag
    function getAnnotationScores() {
        let descEl = null;
        for (const t of document.querySelectorAll('.ant-descriptions-title')) {
            if (t.textContent.includes('标注结果')) {
                descEl = t.closest('.ant-descriptions');
                break;
            }
        }
        if (!descEl) return null;

        const tagCount = descEl.querySelectorAll('.ant-tag').length;
        const items = [];

        for (const labelEl of descEl.querySelectorAll('.ant-descriptions-item-label')) {
            const v = parseFloat(labelEl.textContent.trim());
            if (isNaN(v)) continue;
            // label 和 content 是同一 ant-descriptions-item-container 的兄弟节点
            const container = labelEl.closest('.ant-descriptions-item-container')
                           || labelEl.parentElement;
            const count = container
                ? container.querySelectorAll('.ant-descriptions-item-content .ant-tag').length
                : 1;
            items.push({ score: v, count: Math.max(count, 1) });
        }

        if (items.length === 0) {
            return { total: null, tagCount, scores: [] };
        }

        const realTagCount = items.reduce((s, i) => s + i.count, 0);
        const total = Math.round(
            items.reduce((s, i) => s + i.score * i.count, 0) * 10000
        ) / 10000;
        return { total, tagCount: realTagCount, scores: items };
    }

    // ── 读取"总分"输入框的当前值 ──────────────────────────────────────────
    function getScoreFieldValue() {
        for (const q of document.querySelectorAll('[class*="question___"]')) {
            const title = q.querySelector('[class*="title___"]');
            if (title && /总分|score/i.test(title.textContent)) {
                const inp = q.querySelector('textarea, [class*="input___"]');
                if (inp) return inp.value.trim();
            }
        }
        return null;
    }

    // ── 验证 "a+b+c=total" 格式的总分字符串 ──────────────────────────────
    // 返回 { ok, computed, stated, msg }
    function validateScoreString(str) {
        if (!str) return null;
        const eq = str.lastIndexOf('=');
        if (eq < 0) return { ok: false, msg: '无等号' };

        const stated = parseFloat(str.slice(eq + 1));
        if (isNaN(stated)) return { ok: false, msg: `等号右侧非数字: "${str.slice(eq + 1)}"` };

        let computed = 0;
        for (const p of str.slice(0, eq).split('+')) {
            const v = parseFloat(p.trim());
            if (isNaN(v)) return { ok: false, msg: `部分非数字: "${p.trim()}"` };
            computed += v;
        }
        computed = Math.round(computed * 10000) / 10000;

        const ok = Math.abs(computed - stated) < 0.00005;
        return { ok, computed, stated, msg: ok ? null : `左侧之和 ${computed} ≠ 右侧 ${stated}` };
    }

    // ── UI 面板（右下角，不与 ilabel_physics.js 的左下角面板重叠）────────
    const panel = document.createElement('div');
    panel.style.cssText = `
        position:fixed; bottom:80px; right:20px;
        background:rgba(10,20,50,0.93); color:#E0E0E0;
        padding:10px 14px; border-radius:8px; font-family:monospace;
        font-size:12px; z-index:999998;
        box-shadow:0 4px 16px rgba(0,0,0,0.6); min-width:240px; max-width:360px;
        border:1px solid #1a3a6a; line-height:1.7;
    `;
    document.body.appendChild(panel);

    const _ok   = s => `<span style="color:#2ecc71">✓ ${s}</span>`;
    const _err  = s => `<span style="color:#e74c3c">✗ ${s}</span>`;
    const _grey = s => `<span style="color:#888">${s}</span>`;

    function render() {
        const subQId  = getSubQuestionId();
        const nerInfo = getAnnotationScores();
        const sfVal   = getScoreFieldValue();
        const valid   = sfVal ? validateScoreString(sfVal) : null;

        // 划词总分行
        let nerLine;
        if (!nerInfo) {
            nerLine = _grey('标注结果区域未找到');
        } else if (nerInfo.total === null) {
            nerLine = `<span style="color:#F9CA24">${nerInfo.tagCount} 条标注（分值不可读）</span>`;
        } else {
            nerLine = _ok(`${nerInfo.total} 分（${nerInfo.tagCount} 条）`);
        }

        // 检查①：总分栏算式
        let check1;
        if (!valid) {
            check1 = _grey('总分栏为空');
        } else if (valid.ok) {
            check1 = _ok(`算式正确（= ${valid.stated}）`);
        } else {
            check1 = _err(`算式错误：${valid.msg}`);
        }

        // 检查②：划词总分 vs 总分栏
        let check2 = '';
        const nerTotal = nerInfo?.total;
        if (nerTotal !== null && nerTotal !== undefined && valid) {
            if (Math.abs(nerTotal - valid.stated) < 0.00005) {
                check2 = _ok(`划词总分 = 总分栏（${nerTotal}）`);
            } else {
                check2 = _err(`划词总分 ${nerTotal} ≠ 总分栏 ${valid.stated}`);
            }
        }

        panel.innerHTML = `
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;">
                <b style="color:white;font-size:13px;">📊 分值检查</b>
                <span style="color:#7DF9FF;font-size:11px;">${subQId || _grey('—')}</span>
            </div>
            <hr style="border-color:#1a3a6a;margin:3px 0 5px;">
            <div><span style="color:#AAA">划词总分：</span>${nerLine}</div>
            <div><span style="color:#AAA">总&nbsp;分&nbsp;栏：</span>${sfVal ? `<span style="color:white">${sfVal}</span>` : _grey('—')}</div>
            <hr style="border-color:#1a3a6a;margin:5px 0 3px;">
            <div>${check1}</div>
            ${check2 ? `<div>${check2}</div>` : ''}
        `;
    }

    // ── 启动：DOM 变化时防抖重渲染 ──────────────────────────────────────
    const obs = new MutationObserver(() => {
        clearTimeout(obs._t);
        obs._t = setTimeout(render, 500);
    });
    obs.observe(document.body, { subtree: true, childList: true, characterData: true });
    setTimeout(render, 1500);

})();
