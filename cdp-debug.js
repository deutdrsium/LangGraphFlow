/**
 * cdp-debug.js  —  iLabel 物理划词标注调试工具
 * 用法：node cdp-debug.js [命令]
 *
 * 命令：
 *   inspect   检查当前 DOM 状态（默认）
 *   popup     检查当前可见的标注弹窗结构
 *   console   持续监听 console 输出（Ctrl+C 退出）
 *   eval      在浏览器执行自定义 JS（交互输入）
 */

'use strict';

const http  = require('http');
const https = require('https');
const readline = require('readline');

// ─── CDP 客户端（基于 Node 22 内置 WebSocket）────────────────────────────────
class CDP {
    constructor(wsUrl) {
        this._ws  = new WebSocket(wsUrl);
        this._id  = 0;
        this._cbs = new Map();    // id → {resolve, reject}
        this._evs = new Map();    // method → handler[]
        this._ready = new Promise(res => this._ws.addEventListener('open', res));
        this._ws.addEventListener('message', ({ data }) => {
            const msg = JSON.parse(data);
            if (msg.id != null) {
                const cb = this._cbs.get(msg.id);
                if (cb) { this._cbs.delete(msg.id); cb(msg); }
            } else if (msg.method) {
                (this._evs.get(msg.method) || []).forEach(h => h(msg.params));
            }
        });
    }

    get ready() { return this._ready; }

    send(method, params = {}) {
        return new Promise((resolve, reject) => {
            const id = ++this._id;
            this._cbs.set(id, msg => msg.error ? reject(new Error(msg.error.message)) : resolve(msg.result));
            this._ws.send(JSON.stringify({ id, method, params }));
        });
    }

    on(method, handler) {
        if (!this._evs.has(method)) this._evs.set(method, []);
        this._evs.get(method).push(handler);
    }

    eval(expr) {
        return this.send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: false })
            .then(r => r.result?.value);
    }

    close() { this._ws.close(); }
}

// ─── HTTP helper ─────────────────────────────────────────────────────────────
function httpGet(url) {
    return new Promise((res, rej) => {
        (url.startsWith('https') ? https : http).get(url, r => {
            let d = ''; r.on('data', c => d += c); r.on('end', () => { try { res(JSON.parse(d)); } catch(e) { rej(e); } });
        }).on('error', rej);
    });
}

// ─── 找目标标签页 ─────────────────────────────────────────────────────────────
async function findTarget(keyword = 'ilabel') {
    let targets;
    try {
        targets = await httpGet('http://localhost:9222/json');
    } catch {
        console.error('\n❌  无法连接到 localhost:9222。请先确认：');
        console.error('   1. Chrome 已用 --remote-debugging-port=9222 启动');
        console.error('   2. ilabel.alipay.com 标签页已打开');
        process.exit(1);
    }
    const t = targets.find(x => x.type === 'page' && x.url?.includes(keyword));
    if (!t) {
        console.error('\n❌  未找到包含 "' + keyword + '" 的标签页。当前标签页：');
        targets.filter(x => x.type === 'page').forEach(x => console.error('   - ' + x.title + '\n     ' + x.url));
        process.exit(1);
    }
    return t;
}

// ─── 命令：inspect（默认）────────────────────────────────────────────────────
async function cmdInspect(cdp) {
    const info = await cdp.eval(`(function(){
        function q(s){return document.querySelector(s);}
        function qa(s){return [...document.querySelectorAll(s)];}

        // 子题进度
        let subQ = null;
        const tw = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
        let tn;
        while(tn = tw.nextNode()){
            const m = tn.nodeValue.trim().match(/标注中[（(](\\d+)\\s*[\\/／]\\s*(\\d+)[）)]/);
            if(m){ subQ = m[0]; break; }
        }

        // NER 容器
        const ner = q('[id$="-ner"]');
        const spans = qa('.text-label-letter');

        // 可见 input
        const inputs = qa('input:not([type="hidden"])').filter(i=>i.offsetParent).map(i=>({
            type:i.type, value:i.value.slice(0,40), placeholder:i.placeholder,
            cls:i.className.slice(0,80), parentCls:i.parentElement?.className?.slice(0,80)
        }));

        // 关闭按钮
        const closeBtns = qa('button,[class*="btn"],[class*="close"]').filter(b=>/关闭|close|确定/i.test(b.textContent.trim())).map(b=>({
            text:b.textContent.trim(), cls:b.className.slice(0,80), visible:!!b.offsetParent
        }));

        // textSrc 内容
        const srcs = qa('[class*="textSrc"] > div').map(e=>e.innerText.trim().slice(0,120)).filter(Boolean);

        return JSON.stringify({ subQ, nerContainer:!!ner, nerSpans:spans.length, inputs, closeBtns, textSrc:srcs }, null, 2);
    })()`);
    console.log('\n=== DOM 状态 ===\n');
    try { console.log(JSON.parse(info)); } catch { console.log(info); }
}

// ─── 命令：popup ─────────────────────────────────────────────────────────────
async function cmdPopup(cdp) {
    const info = await cdp.eval(`(function(){
        function qa(s){return [...document.querySelectorAll(s)];}
        // 找所有可见的浮层（宽 < 800, 高 < 600）
        const floats = qa('*').filter(el => {
            if (!el.offsetParent) return false;
            const r = el.getBoundingClientRect();
            if (r.width < 30 || r.height < 20 || r.width > 800 || r.height > 600) return false;
            // 必须包含 input 或关闭按钮
            const hasInput = !!el.querySelector('input');
            const hasClose = [...el.querySelectorAll('button,[class*="btn"]')].some(b=>/关闭|close|确定/i.test(b.textContent));
            return hasInput || hasClose;
        }).map(el => ({
            tag: el.tagName,
            cls: el.className.slice(0,100),
            rect: {w:Math.round(el.getBoundingClientRect().width), h:Math.round(el.getBoundingClientRect().height)},
            html: el.innerHTML.slice(0,400)
        }));
        return JSON.stringify(floats, null, 2);
    })()`);
    console.log('\n=== 弹窗浮层检测 ===\n');
    if (!info || info === '[]') {
        console.log('⚠  未检测到弹窗。请先手动触发标注弹窗，然后再运行此命令。');
    } else {
        try { console.log(JSON.parse(info)); } catch { console.log(info); }
    }
}

// ─── 命令：console ────────────────────────────────────────────────────────────
async function cmdConsole(cdp) {
    await cdp.send('Runtime.enable');
    await cdp.send('Log.enable');

    console.log('\n📡  监听 console 输出（Ctrl+C 退出）...\n');

    cdp.on('Runtime.consoleAPICalled', ({ type, args }) => {
        const txt = args.map(a => a.value ?? a.description ?? JSON.stringify(a.preview) ?? '').join(' ');
        const ts = new Date().toLocaleTimeString();
        const icon = { log:'📋', warn:'⚠', error:'❌', info:'ℹ' }[type] || '▶';
        console.log(`[${ts}] ${icon} ${txt}`);
    });

    cdp.on('Log.entryAdded', ({ entry }) => {
        console.log(`[${entry.source}/${entry.level}] ${entry.text}`);
    });

    // keep alive
    await new Promise(() => {});
}

// ─── 命令：eval ───────────────────────────────────────────────────────────────
async function cmdEval(cdp) {
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    console.log('\n💻  在浏览器执行 JS（空行提交，exit 退出）：\n');
    let buf = [];
    rl.on('line', async line => {
        if (line === 'exit') { rl.close(); cdp.close(); return; }
        if (line === '') {
            if (!buf.length) return;
            const expr = buf.join('\n'); buf = [];
            try {
                const r = await cdp.send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true });
                if (r.exceptionDetails) console.error('❌ ', r.exceptionDetails.text);
                else console.log('→', r.result?.value ?? r.result?.description);
            } catch(e) { console.error('❌ ', e.message); }
        } else {
            buf.push(line);
        }
    });
    await new Promise(res => rl.on('close', res));
}

// ─── main ─────────────────────────────────────────────────────────────────────
(async () => {
    const cmd = process.argv[2] || 'inspect';
    const target = await findTarget('ilabel');
    console.log(`✅  已连接: ${target.title}`);
    const cdp = new CDP(target.webSocketDebuggerUrl);
    await cdp.ready;

    switch (cmd) {
        case 'inspect': await cmdInspect(cdp); cdp.close(); break;
        case 'popup':   await cmdPopup(cdp);   cdp.close(); break;
        case 'console': await cmdConsole(cdp); break; // 持续运行
        case 'eval':    await cmdEval(cdp);    break;
        default:
            console.log('未知命令。可用：inspect | popup | console | eval');
            cdp.close();
    }
})();
