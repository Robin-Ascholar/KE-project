(() => {
    'use strict';

    const SESSION_STORAGE_KEY = 'harry-potter-linearrag.sessions.v1';
    const CONFIG_STORAGE_KEY = 'harry-potter-linearrag.config.v1';
    const ACTIVITY_STORAGE_KEY = 'harry-potter-linearrag.activities.v1';
    const MAX_SESSIONS = 20;

    const SUGGESTED_QUESTIONS = [
        { category: '人物称号', icon: 'user-round-search', question: 'What nickname is Harry Potter known by?' },
        { category: '魔法物品', icon: 'wand-sparkles', question: 'Who arranged for Harry to get the Nimbus Two Thousand?' },
        { category: '地点机关', icon: 'door-open', question: 'How do you open the wall behind the Leaky Cauldron?' },
        { category: '故事情节', icon: 'book-open-text', question: 'What did Harry pull from the Sorting Hat to fight the Basilisk?' },
        { category: '人物关系', icon: 'users-round', question: 'Who are the four creators of the Marauder\'s Map?' },
        { category: '因果推理', icon: 'route', question: 'Why was Harry able to survive Voldemort\'s killing curse as a baby?' },
    ];

    class ApiError extends Error {
        constructor(message, status = 0, body = null) {
            super(message);
            this.name = 'ApiError';
            this.status = status;
            this.body = body;
        }
    }

    class LinearRagWorkbench {
        constructor() {
            this.elements = this.collectElements();
            this.config = this.readStorage(CONFIG_STORAGE_KEY, { topK: 8, csvPath: '' });
            this.sessions = this.readStorage(SESSION_STORAGE_KEY, []);
            this.activities = this.readStorage(ACTIVITY_STORAGE_KEY, []);
            this.activeSessionId = null;
            this.activeView = 'query';
            this.activeDetailTab = 'evidence';
            this.latestResult = null;
            this.status = null;
            this.isSending = false;
            this.isGraphBusy = false;
            this.statusTimer = null;
            this.graphTransform = { x: 0, y: 0, scale: 1 };
        }

        collectElements() {
            const ids = [
                'sidebar', 'mobileScrim', 'newSessionButton', 'historyCount', 'historyList',
                'sidebarStatusButton', 'sidebarStatusDot', 'sidebarStatusText', 'sidebarStatusAddress',
                'pageTitle', 'pageSubtitle', 'mobileMenuButton', 'topKMinus', 'topKInput', 'topKPlus',
                'applyTopKButton', 'topbarStatusButton', 'topbarStatusDot', 'topbarStatusText',
                'mobileTopKMinus', 'mobileTopKInput', 'mobileTopKPlus', 'mobileApplyTopKButton',
                'detailsToggle', 'queryView', 'graphView', 'queryLayout', 'datasetPathLabel', 'openGraphButton',
                'conversation', 'questionForm', 'questionInput', 'sendButton', 'composerState',
                'detailsPanel', 'closeDetailsButton', 'downloadResultButton', 'resultTimestamp',
                'evidenceCount', 'graphNodeCount', 'evidencePane', 'knowledgeGraphPane', 'diagnosticsPane', 'rawPane', 'graphOperationState',
                'csvPathInput', 'resetCsvButton', 'buildGraphButton', 'refreshServiceButton',
                'backendInlineState', 'backendInlineText', 'apiBaseLabel', 'timeoutLabel',
                'clearActivityButton', 'activityList', 'openClearModalButton', 'clearGraphModal',
                'clearConfirmCheckbox', 'cancelClearButton', 'confirmClearButton', 'toastRegion',
            ];
            return Object.fromEntries(ids.map((id) => [id, document.getElementById(id)]));
        }

        init() {
            const initialView = window.location.hash === '#graph' ? 'graph' : 'query';
            this.bindEvents();
            this.restoreState();
            this.renderAll();
            this.switchView(initialView, { updateUrl: false });
            this.loadStatus({ syncTopK: true });
            this.statusTimer = window.setInterval(() => this.loadStatus({ silent: true }), 20000);
            this.refreshIcons();
        }

        bindEvents() {
            const e = this.elements;

            document.querySelectorAll('[data-view]').forEach((button) => {
                button.addEventListener('click', () => this.switchView(button.dataset.view));
            });

            document.querySelectorAll('[data-detail-tab]').forEach((button) => {
                button.addEventListener('click', () => this.switchDetailTab(button.dataset.detailTab));
            });

            e.newSessionButton.addEventListener('click', () => this.createSession(true));
            e.openGraphButton.addEventListener('click', () => this.switchView('graph'));
            e.mobileMenuButton.addEventListener('click', () => this.openMobileSidebar());
            e.mobileScrim.addEventListener('click', () => this.closeOverlays());
            e.detailsToggle.addEventListener('click', () => this.openDetails());
            e.closeDetailsButton.addEventListener('click', () => this.closeDetails());

            e.questionForm.addEventListener('submit', (event) => {
                event.preventDefault();
                this.sendQuestion();
            });
            e.questionInput.addEventListener('input', () => {
                this.resizeQuestionInput();
                this.updateSendButton();
            });
            e.questionInput.addEventListener('keydown', (event) => {
                if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
                    event.preventDefault();
                    this.sendQuestion();
                }
            });

            e.historyList.addEventListener('click', (event) => this.handleHistoryClick(event));
            e.conversation.addEventListener('click', (event) => this.handleConversationClick(event));

            e.topKMinus.addEventListener('click', () => this.bumpTopK(-1));
            e.topKPlus.addEventListener('click', () => this.bumpTopK(1));
            e.topKInput.addEventListener('change', () => this.setTopK(e.topKInput.value));
            e.applyTopKButton.addEventListener('click', () => this.applyTopK());
            e.mobileTopKMinus.addEventListener('click', () => this.bumpTopK(-1));
            e.mobileTopKPlus.addEventListener('click', () => this.bumpTopK(1));
            e.mobileTopKInput.addEventListener('change', () => this.setTopK(e.mobileTopKInput.value));
            e.mobileApplyTopKButton.addEventListener('click', () => this.applyTopK());

            e.sidebarStatusButton.addEventListener('click', () => this.loadStatus());
            e.topbarStatusButton.addEventListener('click', () => this.loadStatus());
            e.refreshServiceButton.addEventListener('click', () => this.loadStatus());

            e.csvPathInput.addEventListener('change', () => this.saveCsvPath());
            e.resetCsvButton.addEventListener('click', () => this.resetCsvPath());
            e.buildGraphButton.addEventListener('click', () => this.buildGraph());
            e.clearActivityButton.addEventListener('click', () => this.clearActivities());
            e.openClearModalButton.addEventListener('click', () => this.openClearModal());
            e.cancelClearButton.addEventListener('click', () => this.closeClearModal());
            e.clearConfirmCheckbox.addEventListener('change', () => {
                e.confirmClearButton.disabled = !e.clearConfirmCheckbox.checked;
            });
            e.confirmClearButton.addEventListener('click', () => this.clearGraph());
            e.clearGraphModal.addEventListener('click', (event) => {
                if (event.target === e.clearGraphModal) this.closeClearModal();
            });

            e.downloadResultButton.addEventListener('click', () => this.downloadLatestResult());
            e.knowledgeGraphPane.addEventListener('click', (event) => this.handleGraphClick(event));

            document.addEventListener('keydown', (event) => {
                if (event.key === 'Escape') this.closeOverlays();
            });
            window.addEventListener('hashchange', () => {
                this.switchView(window.location.hash === '#graph' ? 'graph' : 'query', { updateUrl: false });
            });
        }

        restoreState() {
            const validSessions = Array.isArray(this.sessions)
                ? this.sessions.filter((session) => session && Array.isArray(session.turns))
                : [];
            // 清理上次刷新/关闭页面时遗留的 pending 状态，避免永远显示“正在检索”
            validSessions.forEach((session) => {
                session.turns.forEach((turn) => {
                    if (turn.pending) {
                        delete turn.pending;
                        if (!turn.result && !turn.error) {
                            turn.error = { message: '请求中断（页面已刷新）', status: 0, body: null };
                        }
                    }
                });
            });
            this.sessions = validSessions.slice(0, MAX_SESSIONS);
            this.config.topK = this.clampTopK(this.config.topK || 8);
            this.elements.topKInput.value = String(this.config.topK);
            this.elements.mobileTopKInput.value = String(this.config.topK);

            if (this.sessions.length) {
                this.activeSessionId = this.sessions[0].id;
                this.latestResult = this.lastResult(this.sessions[0]);
            } else {
                this.createSession(false);
            }
        }

        renderAll() {
            this.renderHistory();
            this.renderConversation();
            this.renderDetails(this.latestResult);
            this.renderActivities();
            this.updateSendButton();
        }

        createSession(focusInput = false) {
            const current = this.activeSession();
            if (current && current.turns.length === 0) {
                this.activeSessionId = current.id;
            } else {
                const session = {
                    id: this.makeId(),
                    title: '新问答',
                    createdAt: new Date().toISOString(),
                    updatedAt: new Date().toISOString(),
                    turns: [],
                };
                this.sessions.unshift(session);
                this.sessions = this.sessions.slice(0, MAX_SESSIONS);
                this.activeSessionId = session.id;
            }

            this.latestResult = null;
            this.switchView('query');
            this.renderHistory();
            this.renderConversation();
            this.renderDetails(null);
            this.persistSessions();
            if (focusInput) this.elements.questionInput.focus();
        }

        activeSession() {
            return this.sessions.find((session) => session.id === this.activeSessionId) || null;
        }

        lastResult(session) {
            if (!session) return null;
            for (let index = session.turns.length - 1; index >= 0; index -= 1) {
                if (session.turns[index].result) return session.turns[index].result;
            }
            return null;
        }

        handleHistoryClick(event) {
            const deleteButton = event.target.closest('[data-delete-session]');
            if (deleteButton) {
                event.stopPropagation();
                this.deleteSession(deleteButton.dataset.deleteSession);
                return;
            }

            const openButton = event.target.closest('[data-open-session]');
            if (!openButton) return;
            const session = this.sessions.find((item) => item.id === openButton.dataset.openSession);
            if (!session) return;
            this.activeSessionId = session.id;
            this.latestResult = this.lastResult(session);
            this.switchView('query');
            this.renderHistory();
            this.renderConversation();
            this.renderDetails(this.latestResult);
        }

        deleteSession(sessionId) {
            this.sessions = this.sessions.filter((session) => session.id !== sessionId);
            if (this.activeSessionId === sessionId) {
                this.activeSessionId = this.sessions[0]?.id || null;
                if (!this.activeSessionId) this.createSession(false);
                this.latestResult = this.lastResult(this.activeSession());
            }
            this.persistSessions();
            this.renderHistory();
            this.renderConversation();
            this.renderDetails(this.latestResult);
        }

        renderHistory() {
            const visibleSessions = this.sessions.filter((session) => session.turns.length > 0);
            this.elements.historyCount.textContent = String(visibleSessions.length);
            if (!visibleSessions.length) {
                this.elements.historyList.innerHTML = '<div class="history-empty">暂无问答记录</div>';
                return;
            }

            this.elements.historyList.innerHTML = visibleSessions.map((session) => `
                <div class="history-row ${session.id === this.activeSessionId ? 'is-active' : ''}">
                    <button class="history-open" type="button" data-open-session="${this.escapeHtml(session.id)}">
                        <strong>${this.escapeHtml(session.title)}</strong>
                        <small>${this.escapeHtml(this.formatRelativeTime(session.updatedAt))}</small>
                    </button>
                    <button class="history-delete" type="button" data-delete-session="${this.escapeHtml(session.id)}" title="删除记录" aria-label="删除记录">
                        <i data-lucide="x" aria-hidden="true"></i>
                    </button>
                </div>
            `).join('');
            this.refreshIcons();
        }

        renderConversation() {
            const session = this.activeSession();
            if (!session || session.turns.length === 0) {
                this.elements.conversation.innerHTML = `
                    <div class="welcome">
                        <section class="welcome-stage">
                            <div class="welcome-stage-topline">
                                <span class="welcome-kicker"><i data-lucide="sparkles" aria-hidden="true"></i> TIANJIN UNIVERSITY · AI</span>
                                <span class="welcome-live-status" id="welcomeLiveStatus">
                                    <i class="status-dot ${this.status?.backend?.reachable ? 'is-online' : 'is-offline'}"></i>
                                    <span>${this.status?.backend?.reachable ? 'LinearRAG 已就绪' : '等待加载索引'}</span>
                                </span>
                            </div>
                            <div class="welcome-title-row">
                                <div class="welcome-mark"><i data-lucide="network" aria-hidden="true"></i></div>
                                <div>
                                    <h2>Harry Potter 原著问答</h2>
                                    <p>LINEARRAG · SEVEN-BOOK KNOWLEDGE BASE</p>
                                </div>
                            </div>
                            <div class="schema-visual" aria-label="LinearRAG 检索结构">
                                <span class="schema-chip" title="Book">
                                    <i data-lucide="library-big" aria-hidden="true"></i>书册
                                </span>
                                <span class="schema-arrow" aria-hidden="true"><i data-lucide="arrow-right"></i></span>
                                <span class="schema-chip schema-chip-core" title="Chapter">
                                    <i data-lucide="book-open" aria-hidden="true"></i>章节
                                </span>
                                <span class="schema-arrow" aria-hidden="true"><i data-lucide="arrow-right"></i></span>
                                <span class="schema-chip schema-chip-effect" title="Entity">
                                    <i data-lucide="users" aria-hidden="true"></i>实体
                                </span>
                                <span class="schema-chip schema-chip-cause" title="Sentence">
                                    <i data-lucide="text-search" aria-hidden="true"></i>句子
                                </span>
                                <span class="schema-chip schema-chip-chunk" title="Passage">
                                    <i data-lucide="scan-text" aria-hidden="true"></i>Passage
                                </span>
                            </div>
                            <div class="welcome-facts">
                                <span><i data-lucide="file-spreadsheet" aria-hidden="true"></i><strong id="welcomeDataset">${this.escapeHtml(this.config.csvPath || this.status?.defaultCsvPath || 'harrypotter')}</strong></span>
                                <span><i data-lucide="sliders-horizontal" aria-hidden="true"></i><strong id="welcomeTopK">Top-K ${this.config.topK}</strong></span>
                                <span><i data-lucide="database" aria-hidden="true"></i><strong>GraphML</strong></span>
                            </div>
                        </section>
                        <div class="prompt-section-heading">
                            <div><span>QUERY STARTERS</span><h3>示例问题</h3></div>
                            <span class="prompt-count">${SUGGESTED_QUESTIONS.length}</span>
                        </div>
                        <div class="prompt-grid">
                            ${SUGGESTED_QUESTIONS.map((item) => `
                                <button class="prompt-button" type="button" data-question="${this.escapeHtml(item.question)}">
                                    <i data-lucide="${item.icon}" aria-hidden="true"></i>
                                    <span class="prompt-copy"><small>${this.escapeHtml(item.category)}</small><strong>${this.escapeHtml(item.question)}</strong></span>
                                    <i class="prompt-arrow" data-lucide="arrow-up-right" aria-hidden="true"></i>
                                </button>
                            `).join('')}
                        </div>
                    </div>
                `;
                this.refreshIcons();
                return;
            }

            this.elements.conversation.innerHTML = session.turns.map((turn, index) => {
                let answer = turn.result?.answer || turn.error?.message || '';
                if (!answer && turn.result && !turn.pending) answer = '（后端未返回答案文本）';
                const backend = turn.result ? this.backendPayload(turn.result) : {};
                const route = backend.route || '';
                const elapsed = turn.result?.elapsed_ms;
                const status = turn.error?.status;
                const answerBody = turn.pending
                    ? `<div class="loading-line"><span class="loading-dots"><i></i><i></i><i></i></span><span>正在检索</span></div>`
                    : `<div class="answer-text">${this.escapeHtml(answer)}</div>`;

                return `
                    <article class="turn" data-turn-index="${index}">
                        <div class="message message-user">
                            <div class="user-bubble">${this.escapeHtml(turn.question)}</div>
                        </div>
                        <div class="message message-assistant ${turn.error ? 'message-error' : ''}">
                            <div class="assistant-mark"><i data-lucide="${turn.error ? 'circle-alert' : 'network'}" aria-hidden="true"></i></div>
                            <div class="assistant-content">
                                ${turn.pending ? '' : `<div class="assistant-heading"><strong>${turn.error ? '请求异常' : 'LinearRAG 回答'}</strong><span>${this.escapeHtml(this.formatDateTime(turn.createdAt))}</span></div>`}
                                ${answerBody}
                                ${turn.pending ? '' : `
                                    <div class="answer-footer">
                                        ${route ? `<span class="route-badge">${this.escapeHtml(route)}</span>` : ''}
                                        ${Number.isFinite(elapsed) ? `<span class="elapsed-badge">${elapsed} ms</span>` : ''}
                                        ${status ? `<span class="elapsed-badge">HTTP ${status}</span>` : ''}
                                        ${turn.result ? `
                                            <button class="icon-button" type="button" data-copy-answer="${index}" title="复制答案" aria-label="复制答案">
                                                <i data-lucide="copy" aria-hidden="true"></i>
                                            </button>
                                            <button class="icon-button" type="button" data-inspect-answer="${index}" title="查看检索详情" aria-label="查看检索详情">
                                                <i data-lucide="panel-right-open" aria-hidden="true"></i>
                                            </button>
                                            <button class="icon-button" type="button" data-view-graph="${index}" title="查看知识图谱" aria-label="查看知识图谱">
                                                <i data-lucide="share-2" aria-hidden="true"></i>
                                            </button>
                                        ` : ''}
                                    </div>
                                `}
                            </div>
                        </div>
                    </article>
                `;
            }).join('');
            this.refreshIcons();
            requestAnimationFrame(() => {
                this.elements.conversation.scrollTop = this.elements.conversation.scrollHeight;
            });
        }

        handleConversationClick(event) {
            const prompt = event.target.closest('[data-question]');
            if (prompt) {
                this.elements.questionInput.value = prompt.dataset.question;
                this.resizeQuestionInput();
                this.updateSendButton();
                this.elements.questionInput.focus();
                return;
            }

            const copyButton = event.target.closest('[data-copy-answer]');
            if (copyButton) {
                const turn = this.activeSession()?.turns[Number(copyButton.dataset.copyAnswer)];
                if (turn?.result?.answer) this.copyText(turn.result.answer, '答案已复制');
                return;
            }

            const graphButton = event.target.closest('[data-view-graph]');
            if (graphButton) {
                const turn = this.activeSession()?.turns[Number(graphButton.dataset.viewGraph)];
                if (turn?.result) {
                    this.latestResult = turn.result;
                    this.renderDetails(turn.result);
                    this.switchDetailTab('knowledge-graph');
                    this.openDetails();
                }
                return;
            }

            const inspectButton = event.target.closest('[data-inspect-answer]');
            if (inspectButton) {
                const turn = this.activeSession()?.turns[Number(inspectButton.dataset.inspectAnswer)];
                if (turn?.result) {
                    this.latestResult = turn.result;
                    this.renderDetails(turn.result);
                    this.openDetails();
                }
            }
        }

        async sendQuestion() {
            const question = this.elements.questionInput.value.trim();
            if (!question || this.isSending) return;

            let session = this.activeSession();
            if (!session) {
                this.createSession(false);
                session = this.activeSession();
            }

            const turn = {
                id: this.makeId(),
                question,
                createdAt: new Date().toISOString(),
                pending: true,
            };
            session.turns.push(turn);
            if (session.turns.length === 1) session.title = question.slice(0, 32);
            session.updatedAt = new Date().toISOString();
            this.elements.questionInput.value = '';
            this.resizeQuestionInput();
            this.isSending = true;
            this.elements.composerState.textContent = '正在检索';
            this.updateSendButton();
            this.renderHistory();
            this.renderConversation();

            try {
                const result = await this.requestJson('/api/chat', {
                    method: 'POST',
                    body: JSON.stringify({ question, top_k: this.config.topK }),
                });
                turn.result = result;
                this.latestResult = result;
                this.renderDetails(result);
                this.elements.composerState.textContent = '检索完成';
            } catch (error) {
                turn.error = {
                    message: error.message || '请求失败',
                    status: error.status || 0,
                    body: error.body || null,
                };
                this.elements.composerState.textContent = '请求失败';
                if (error.status === 409) {
                    this.toast('LinearRAG 索引尚未加载，请先在「索引管理」中加载', 'warning');
                }
                if (error.status === 502 || error.status === 503) this.loadStatus({ silent: true });
            } finally {
                delete turn.pending;
                session.updatedAt = new Date().toISOString();
                this.isSending = false;
                this.persistSessions();
                this.renderHistory();
                this.renderConversation();
                this.updateSendButton();
                this.elements.questionInput.focus();
            }
        }

        renderDetails(result) {
            this.latestResult = result || null;
            this.elements.downloadResultButton.disabled = !result;
            this.elements.resultTimestamp.textContent = result
                ? `响应耗时 ${Number.isFinite(result.elapsed_ms) ? `${result.elapsed_ms} ms` : '未提供'}`
                : '尚无结果';

            if (!result) {
                const empty = this.detailEmpty('scan-search', '问答完成后显示后端返回内容');
                this.elements.evidencePane.innerHTML = empty;
                this.elements.diagnosticsPane.innerHTML = empty;
                this.elements.rawPane.innerHTML = empty;
                this.elements.knowledgeGraphPane.innerHTML = this.graphEmpty('完成一次问答后，将展示本次检索命中的局部知识图谱');
                this.elements.evidenceCount.textContent = '0';
                this.elements.graphNodeCount.textContent = '0';
                this.refreshIcons();
                return;
            }

            const evidence = this.collectEvidence(result);
            this.elements.evidenceCount.textContent = String(evidence.length);
            this.elements.evidencePane.innerHTML = evidence.length
                ? evidence.map((item, index) => `
                    <article class="evidence-item">
                        <div class="evidence-item-head">
                            <strong>${this.escapeHtml(item.source)}</strong>
                            <span>${String(index + 1).padStart(2, '0')}</span>
                        </div>
                        <pre class="evidence-value">${this.escapeHtml(this.formatValue(item.value))}</pre>
                    </article>
                `).join('')
                : this.detailEmpty('inbox', '后端未返回检索证据');

            const graph = result.graph && typeof result.graph === 'object' ? result.graph : null;
            this.elements.graphNodeCount.textContent = String(graph?.nodes?.length || 0);
            this.renderKnowledgeGraph(graph);

            const backend = this.backendPayload(result);
            const routeConfidence = backend.route_confidence;
            const missingSlots = Array.isArray(backend.missing_slots) ? backend.missing_slots : [];
            const queryIr = backend.query_ir && typeof backend.query_ir === 'object' ? backend.query_ir : null;
            this.elements.diagnosticsPane.innerHTML = `
                <div class="diagnostic-block">
                    <h3>执行信息</h3>
                    <dl class="diagnostic-list">
                        <div><dt>检索模式</dt><dd>${this.escapeHtml(backend.route || '未提供')}</dd></div>
                        <div><dt>候选章节数</dt><dd>${routeConfidence === 0 || routeConfidence ? this.escapeHtml(String(routeConfidence)) : '未提供'}</dd></div>
                        <div><dt>Top-K 参数</dt><dd>${missingSlots.length ? this.escapeHtml(missingSlots.join(', ')) : '无'}</dd></div>
                        <div><dt>问答模型</dt><dd>${this.escapeHtml(result.answer_file || backend.answer_file || '未提供')}</dd></div>
                        <div><dt>响应耗时</dt><dd>${Number.isFinite(result.elapsed_ms) ? `${result.elapsed_ms} ms` : '未提供'}</dd></div>
                    </dl>
                </div>
                <div class="diagnostic-block">
                    <h3>LinearRAG 诊断</h3>
                    ${queryIr
                        ? `<pre class="query-ir-json">${this.escapeHtml(JSON.stringify(queryIr, null, 2))}</pre>`
                        : '<div class="detail-empty"><div class="detail-empty-inner"><span>后端未返回检索诊断</span></div></div>'}
                </div>
            `;
            this.elements.rawPane.innerHTML = `<pre class="raw-json">${this.escapeHtml(JSON.stringify(result, null, 2))}</pre>`;
            this.refreshIcons();
        }

        graphEmpty(message) {
            return `<div class="kg-empty"><i data-lucide="share-2" aria-hidden="true"></i><span>${this.escapeHtml(message)}</span></div>`;
        }

        renderKnowledgeGraph(graph) {
            const pane = this.elements.knowledgeGraphPane;
            const nodes = Array.isArray(graph?.nodes) ? graph.nodes : [];
            const edges = Array.isArray(graph?.edges) ? graph.edges : [];
            if (!nodes.length) {
                pane.innerHTML = this.graphEmpty('本次检索没有返回可展示的图谱节点');
                this.refreshIcons();
                return;
            }

            const width = 900;
            const height = 600;
            const nodeMap = new Map();
            const passages = nodes.filter((node) => node.type === 'passage');
            const passageIndex = new Map(passages.map((node, index) => [node.id, index]));
            const edgeMap = new Map();
            edges.forEach((edge) => {
                if (!edgeMap.has(edge.source)) edgeMap.set(edge.source, []);
                if (!edgeMap.has(edge.target)) edgeMap.set(edge.target, []);
                edgeMap.get(edge.source).push(edge.target);
                edgeMap.get(edge.target).push(edge.source);
            });

            nodes.forEach((node, index) => {
                let angle;
                let radius;
                if (node.type === 'passage') {
                    angle = (Math.PI * 2 * (passageIndex.get(node.id) || 0)) / Math.max(1, passages.length) - Math.PI / 2;
                    radius = passages.length === 1 ? 0 : 145;
                } else {
                    const linkedPassage = (edgeMap.get(node.id) || []).find((id) => passageIndex.has(id));
                    const anchor = passageIndex.get(linkedPassage) || 0;
                    const seed = this.stableHash(node.id);
                    angle = (Math.PI * 2 * anchor) / Math.max(1, passages.length) + ((seed % 100) / 100 - 0.5) * 1.35;
                    radius = 235 + (seed % 105);
                }
                nodeMap.set(node.id, {
                    ...node,
                    x: width / 2 + Math.cos(angle) * radius,
                    y: height / 2 + Math.sin(angle) * radius,
                    vx: 0,
                    vy: 0,
                });
            });

            const validEdges = edges.filter((edge) => nodeMap.has(edge.source) && nodeMap.has(edge.target));
            const layoutNodes = [...nodeMap.values()];
            for (let tick = 0; tick < 150; tick += 1) {
                const alpha = 1 - tick / 150;
                for (let i = 0; i < layoutNodes.length; i += 1) {
                    const a = layoutNodes[i];
                    for (let j = i + 1; j < layoutNodes.length; j += 1) {
                        const b = layoutNodes[j];
                        let dx = b.x - a.x;
                        let dy = b.y - a.y;
                        const distance2 = Math.max(80, dx * dx + dy * dy);
                        const force = 1250 * alpha / distance2;
                        const distance = Math.sqrt(distance2);
                        dx /= distance;
                        dy /= distance;
                        a.vx -= dx * force;
                        a.vy -= dy * force;
                        b.vx += dx * force;
                        b.vy += dy * force;
                    }
                }
                validEdges.forEach((edge) => {
                    const a = nodeMap.get(edge.source);
                    const b = nodeMap.get(edge.target);
                    const dx = b.x - a.x;
                    const dy = b.y - a.y;
                    const distance = Math.max(1, Math.hypot(dx, dy));
                    const desired = a.type === 'passage' || b.type === 'passage' ? 105 : 135;
                    const force = (distance - desired) * 0.018 * alpha;
                    a.vx += dx / distance * force;
                    a.vy += dy / distance * force;
                    b.vx -= dx / distance * force;
                    b.vy -= dy / distance * force;
                });
                layoutNodes.forEach((node) => {
                    node.vx += (width / 2 - node.x) * 0.0018 * alpha;
                    node.vy += (height / 2 - node.y) * 0.0018 * alpha;
                    node.vx *= 0.78;
                    node.vy *= 0.78;
                    node.x = Math.max(42, Math.min(width - 42, node.x + node.vx));
                    node.y = Math.max(42, Math.min(height - 42, node.y + node.vy));
                });
            }

            const stats = graph.stats || {};
            pane.innerHTML = `
                <div class="kg-shell">
                    <div class="kg-toolbar">
                        <div>
                            <strong>查询局部图谱</strong>
                            <span>${stats.entities || nodes.filter((n) => n.type === 'entity').length} 个实体 · ${stats.seed_entities || nodes.filter((n) => n.is_seed).length} 个查询种子 · ${stats.relations || validEdges.length} 条关系</span>
                        </div>
                        <div class="kg-actions">
                            <button type="button" data-graph-action="zoom-out" title="缩小" aria-label="缩小"><i data-lucide="minus"></i></button>
                            <button type="button" data-graph-action="reset" title="复位" aria-label="复位"><i data-lucide="maximize-2"></i></button>
                            <button type="button" data-graph-action="zoom-in" title="放大" aria-label="放大"><i data-lucide="plus"></i></button>
                        </div>
                    </div>
                    <div class="kg-canvas-wrap">
                        <svg class="kg-canvas" viewBox="0 0 ${width} ${height}" role="img" aria-label="本次问答检索到的局部知识图谱">
                            <defs>
                                <filter id="kgGlow"><feGaussianBlur stdDeviation="3" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
                            </defs>
                            <g class="kg-viewport">
                                <g class="kg-edges">
                                    ${validEdges.map((edge, index) => {
                                        const a = nodeMap.get(edge.source);
                                        const b = nodeMap.get(edge.target);
                                        return `<line class="${edge.label === 'co-occurs' ? 'is-cooccurrence' : 'is-mention'}" data-edge-index="${index}" x1="${a.x.toFixed(1)}" y1="${a.y.toFixed(1)}" x2="${b.x.toFixed(1)}" y2="${b.y.toFixed(1)}"></line>`;
                                    }).join('')}
                                </g>
                                <g class="kg-nodes">
                                    ${layoutNodes.map((node, nodeIndex) => {
                                        const labelClass = node.type === 'passage' || node.is_seed || nodeIndex < 24 ? ' is-label-visible' : '';
                                        const nodeClass = (node.type === 'passage' ? 'is-passage' : node.is_seed ? 'is-seed' : node.is_direct ? 'is-entity is-direct' : 'is-entity is-expanded') + labelClass;
                                        const radius = node.type === 'passage' ? 15 : node.is_seed ? 12 : 9;
                                        const label = this.truncateLabel(node.label, node.type === 'passage' ? 22 : 18);
                                        const nodeDetail = node.type === 'passage' ? (node.chapter_title || '检索证据段落') : `${node.is_seed ? '查询种子实体' : node.is_direct ? '直接相关实体' : '扩展共现实体'} · 连接 ${Number(node.degree) || 0} 个节点`;
                                        return `<g class="kg-node ${nodeClass}" data-node-id="${this.escapeHtml(node.id)}" transform="translate(${node.x.toFixed(1)} ${node.y.toFixed(1)})" tabindex="0" role="button" aria-label="${this.escapeHtml(node.label)}" data-node-label="${this.escapeHtml(node.label)}" data-node-detail="${this.escapeHtml(nodeDetail)}" onclick="document.getElementById('kgInspector').textContent=this.dataset.nodeLabel+' · '+this.dataset.nodeDetail">
                                            <circle r="${radius}"></circle>
                                            <text y="${radius + 16}" text-anchor="middle">${this.escapeHtml(label)}</text>
                                        </g>`;
                                    }).join('')}
                                </g>
                            </g>
                        </svg>
                        <div class="kg-hint">滚轮缩放 · 拖动画布 · 拖拽节点 · 点击查看详情</div>
                    </div>
                    <div class="kg-footer">
                        <div class="kg-legend"><span><i class="seed"></i>查询种子实体</span><span><i class="entity"></i>相关实体</span><span><i class="passage"></i>检索段落</span><span><b></b>出现 / 共现关系</span></div>
                        <div class="kg-inspector" id="kgInspector">点击节点查看内容</div>
                    </div>
                </div>
            `;
            this.refreshIcons();
            this.installGraphInteractions(nodeMap, validEdges);
        }

        installGraphInteractions(nodeMap, edges) {
            const svg = this.elements.knowledgeGraphPane.querySelector('.kg-canvas');
            const viewport = svg?.querySelector('.kg-viewport');
            if (!svg || !viewport) return;
            this.graphTransform = { x: 0, y: 0, scale: 1 };
            const applyTransform = () => {
                const t = this.graphTransform;
                viewport.setAttribute('transform', `translate(${t.x} ${t.y}) scale(${t.scale})`);
            };
            const updateNode = (id) => {
                const node = nodeMap.get(id);
                const group = svg.querySelector(`[data-node-id="${CSS.escape(id)}"]`);
                if (group) group.setAttribute('transform', `translate(${node.x.toFixed(1)} ${node.y.toFixed(1)})`);
                edges.forEach((edge, index) => {
                    if (edge.source !== id && edge.target !== id) return;
                    const line = svg.querySelector(`[data-edge-index="${index}"]`);
                    const a = nodeMap.get(edge.source);
                    const b = nodeMap.get(edge.target);
                    if (line) {
                        line.setAttribute('x1', a.x.toFixed(1));
                        line.setAttribute('y1', a.y.toFixed(1));
                        line.setAttribute('x2', b.x.toFixed(1));
                        line.setAttribute('y2', b.y.toFixed(1));
                    }
                });
            };
            const svgPoint = (event) => {
                const point = svg.createSVGPoint();
                point.x = event.clientX;
                point.y = event.clientY;
                return point.matrixTransform(svg.getScreenCTM().inverse());
            };
            let interaction = null;
            svg.addEventListener('pointerdown', (event) => {
                const nodeElement = event.target.closest?.('.kg-node');
                const point = svgPoint(event);
                if (nodeElement) {
                    interaction = { type: 'node', id: nodeElement.dataset.nodeId };
                } else {
                    interaction = { type: 'pan', startX: point.x, startY: point.y, originX: this.graphTransform.x, originY: this.graphTransform.y };
                    svg.setPointerCapture(event.pointerId);
                }
            });
            svg.addEventListener('pointermove', (event) => {
                if (!interaction) return;
                const point = svgPoint(event);
                if (interaction.type === 'pan') {
                    this.graphTransform.x = interaction.originX + point.x - interaction.startX;
                    this.graphTransform.y = interaction.originY + point.y - interaction.startY;
                    applyTransform();
                } else {
                    const node = nodeMap.get(interaction.id);
                    node.x = (point.x - this.graphTransform.x) / this.graphTransform.scale;
                    node.y = (point.y - this.graphTransform.y) / this.graphTransform.scale;
                    updateNode(interaction.id);
                }
            });
            svg.querySelectorAll('.kg-node').forEach((nodeElement) => {
                nodeElement.addEventListener('pointerup', () => {
                    if (interaction?.type === 'node') this.selectGraphNode(nodeElement);
                });
            });
            const endInteraction = () => { interaction = null; };
            svg.addEventListener('pointerup', endInteraction);
            svg.addEventListener('pointercancel', endInteraction);
            svg.addEventListener('wheel', (event) => {
                event.preventDefault();
                const factor = event.deltaY < 0 ? 1.12 : 0.89;
                this.graphTransform.scale = Math.max(0.45, Math.min(2.6, this.graphTransform.scale * factor));
                applyTransform();
            }, { passive: false });
            this._applyGraphTransform = (action) => {
                if (action === 'reset') this.graphTransform = { x: 0, y: 0, scale: 1 };
                if (action === 'zoom-in') this.graphTransform.scale = Math.min(2.6, this.graphTransform.scale * 1.2);
                if (action === 'zoom-out') this.graphTransform.scale = Math.max(0.45, this.graphTransform.scale / 1.2);
                applyTransform();
            };
            this._graphNodeMap = nodeMap;
        }

        handleGraphClick(event) {
            const action = event.target.closest('[data-graph-action]');
            if (action) {
                this._applyGraphTransform?.(action.dataset.graphAction);
                return;
            }
            const nodeElement = event.target.closest('.kg-node');
            if (nodeElement) this.selectGraphNode(nodeElement);
        }

        selectGraphNode(nodeElement) {
            if (!nodeElement || !this._graphNodeMap) return;
            const node = this._graphNodeMap.get(nodeElement.dataset.nodeId);
            if (!node) return;
            this.elements.knowledgeGraphPane.querySelectorAll('.kg-node.is-selected').forEach((item) => item.classList.remove('is-selected'));
            nodeElement.classList.add('is-selected');
            const inspector = this.elements.knowledgeGraphPane.querySelector('#kgInspector');
            if (!inspector) return;
            if (node.type === 'passage') {
                inspector.innerHTML = `<strong>${this.escapeHtml(node.label)}</strong><span>${this.escapeHtml(node.chapter_title || '')}</span><p>${this.escapeHtml(this.truncateLabel(node.passage || '', 260))}</p>`;
            } else {
                inspector.innerHTML = `<strong>${this.escapeHtml(node.label)}</strong><span>${node.is_seed ? '查询种子实体' : node.is_direct ? '直接相关实体' : '扩展共现实体'} · 连接 ${Number(node.degree) || 0} 个节点</span>`;
            }
        }

        stableHash(value) {
            let hash = 2166136261;
            for (let index = 0; index < String(value).length; index += 1) {
                hash ^= String(value).charCodeAt(index);
                hash = Math.imul(hash, 16777619);
            }
            return Math.abs(hash >>> 0);
        }

        truncateLabel(value, limit) {
            const text = String(value || '').replace(/\s+/g, ' ').trim();
            return text.length > limit ? `${text.slice(0, limit - 1)}…` : text;
        }


        collectEvidence(result) {
            const items = [];
            this.asList(result.context).forEach((value) => {
                if (value && typeof value === 'object' && value.passage) {
                    const location = value.book
                        ? 'Book ' + value.book + ' · Chapter ' + value.chapter
                        : '检索 Passage';
                    const score = Number.isFinite(Number(value.score))
                        ? ' · score ' + Number(value.score).toFixed(4)
                        : '';
                    items.push({ source: location + score, value: value.passage });
                } else {
                    items.push({ source: '检索 Passage', value });
                }
            });
            this.asList(result.context_raw).forEach((value) => items.push({ source: '原始检索数据', value }));
            return items;
        }

        asList(value) {
            if (value === null || value === undefined || value === '') return [];
            return Array.isArray(value) ? value : [value];
        }

        backendPayload(result) {
            return result && typeof result.raw === 'object' && result.raw !== null ? result.raw : {};
        }

        switchDetailTab(tab) {
            this.activeDetailTab = tab;
            this.elements.queryLayout.classList.toggle('is-graph-expanded', tab === 'knowledge-graph');
            document.querySelectorAll('[data-detail-tab]').forEach((button) => {
                button.classList.toggle('is-active', button.dataset.detailTab === tab);
                button.setAttribute('aria-selected', button.dataset.detailTab === tab ? 'true' : 'false');
            });
            document.querySelectorAll('[data-detail-pane]').forEach((pane) => {
                pane.classList.toggle('is-active', pane.dataset.detailPane === tab);
            });
        }

        switchView(view, { updateUrl = true } = {}) {
            this.activeView = view === 'graph' ? 'graph' : 'query';
            this.elements.queryView.classList.toggle('is-active', this.activeView === 'query');
            this.elements.graphView.classList.toggle('is-active', this.activeView === 'graph');
            document.querySelectorAll('[data-view]').forEach((button) => {
                button.classList.toggle('is-active', button.dataset.view === this.activeView);
            });
            const graph = this.activeView === 'graph';
            this.elements.pageTitle.textContent = graph ? '索引管理' : '问答工作台';
            this.elements.pageSubtitle.textContent = graph ? 'GRAPHML · EMBEDDING · OLLAMA' : 'HARRY POTTER · LINEARRAG · QA';
            if (updateUrl) {
                const targetHash = graph ? '#graph' : '#query';
                if (window.location.hash !== targetHash) window.history.replaceState(null, '', targetHash);
            }
            this.closeMobileSidebar();
            if (graph) this.loadStatus({ silent: true });
        }

        openMobileSidebar() {
            this.elements.sidebar.classList.add('is-open');
            this.syncScrim();
        }

        closeMobileSidebar() {
            this.elements.sidebar.classList.remove('is-open');
            this.syncScrim();
        }

        openDetails() {
            this.elements.detailsPanel.classList.add('is-open');
            this.syncScrim();
        }

        closeDetails() {
            this.elements.detailsPanel.classList.remove('is-open');
            this.syncScrim();
        }

        closeOverlays() {
            this.closeMobileSidebar();
            this.closeDetails();
            this.closeClearModal();
        }

        syncScrim() {
            const active = this.elements.sidebar.classList.contains('is-open')
                || this.elements.detailsPanel.classList.contains('is-open');
            this.elements.mobileScrim.hidden = !active;
        }

        setTopK(value) {
            const next = this.clampTopK(value);
            this.config.topK = next;
            this.elements.topKInput.value = String(next);
            this.elements.mobileTopKInput.value = String(next);
            const welcomeTopK = document.getElementById('welcomeTopK');
            if (welcomeTopK) welcomeTopK.textContent = `Top-K ${next}`;
            this.elements.applyTopKButton.textContent = '应用';
            this.elements.mobileApplyTopKButton.textContent = '应用';
            this.writeStorage(CONFIG_STORAGE_KEY, this.config);
        }

        bumpTopK(delta) {
            this.setTopK(this.config.topK + delta);
        }

        clampTopK(value) {
            const parsed = Number.parseInt(value, 10);
            return Math.min(20, Math.max(1, Number.isFinite(parsed) ? parsed : 8));
        }

        async applyTopK({ silent = false } = {}) {
            // config.topK 已由输入框 change 事件同步，避免移动端应用时误读桌面输入框
            this.setTopK(this.config.topK);
            const button = this.elements.applyTopKButton;
            const mobileButton = this.elements.mobileApplyTopKButton;
            button.disabled = true;
            mobileButton.disabled = true;
            try {
                await this.requestJson('/api/top-k', {
                    method: 'POST',
                    body: JSON.stringify({ top_k: this.config.topK }),
                });
                button.textContent = '已应用';
                mobileButton.textContent = '已应用';
                if (!silent) this.toast(`Top-K 已设置为 ${this.config.topK}`, 'success');
            } catch (error) {
                button.textContent = '重试';
                mobileButton.textContent = '重试';
                if (!silent) this.toast(error.message, 'error');
            } finally {
                button.disabled = false;
                mobileButton.disabled = false;
            }
        }

        saveCsvPath() {
            const path = this.elements.csvPathInput.value.trim();
            this.config.csvPath = path;
            this.elements.datasetPathLabel.textContent = path || this.status?.defaultCsvPath || 'harrypotter';
            const welcomeDataset = document.getElementById('welcomeDataset');
            if (welcomeDataset) welcomeDataset.textContent = path || this.status?.defaultCsvPath || 'harrypotter';
            this.writeStorage(CONFIG_STORAGE_KEY, this.config);
        }

        resetCsvPath() {
            const path = this.status?.defaultCsvPath || 'harrypotter';
            this.elements.csvPathInput.value = path;
            this.saveCsvPath();
        }

        async buildGraph() {
            if (this.isGraphBusy) return;
            const path = this.elements.csvPathInput.value.trim();
            if (!path) {
                this.toast('请输入数据集名称', 'warning');
                this.elements.csvPathInput.focus();
                return;
            }

            this.saveCsvPath();
            this.setGraphBusy(true, '正在加载索引');
            const started = Date.now();
            try {
                const result = await this.requestJson('/api/load-index', {
                    method: 'POST',
                    body: JSON.stringify({ path }),
                });
                this.addActivity({
                    type: 'build',
                    status: 'success',
                    title: '索引加载成功',
                    detail: `${path} · ${result.elapsed_ms ?? Date.now() - started} ms`,
                    response: result,
                });
                this.setGraphBusy(false, '索引已加载', 'success');
                this.toast('图谱与向量索引已加载', 'success');
            } catch (error) {
                this.addActivity({
                    type: 'build',
                    status: 'error',
                    title: '索引加载失败',
                    detail: error.message,
                    response: error.body,
                });
                this.setGraphBusy(false, '加载失败', 'error');
                this.toast(error.message, 'error');
            }
        }

        openClearModal() {
            this.elements.clearConfirmCheckbox.checked = false;
            this.elements.confirmClearButton.disabled = true;
            this.elements.clearGraphModal.classList.add('is-open');
            this.elements.clearGraphModal.setAttribute('aria-hidden', 'false');
        }

        closeClearModal() {
            this.elements.clearGraphModal.classList.remove('is-open');
            this.elements.clearGraphModal.setAttribute('aria-hidden', 'true');
        }

        async clearGraph() {
            if (!this.elements.clearConfirmCheckbox.checked || this.isGraphBusy) return;
            this.closeClearModal();
            this.setGraphBusy(true, '正在卸载运行时');
            try {
                const result = await this.requestJson('/api/unload', {
                    method: 'POST',
                    body: JSON.stringify({ confirm: true }),
                });
                const unloaded = Boolean(result?.result?.unloaded);
                this.addActivity({
                    type: 'clear',
                    status: 'success',
                    title: '运行时已卸载',
                    detail: unloaded ? '已释放内存，磁盘索引保留' : '运行时原本未加载',
                    response: result,
                });
                this.setGraphBusy(false, '运行时已卸载', 'success');
                this.toast('运行时已卸载，磁盘索引仍保留', 'success');
                this.loadStatus({ silent: true });
            } catch (error) {
                this.addActivity({
                    type: 'clear',
                    status: 'error',
                    title: '卸载运行时失败',
                    detail: error.message,
                    response: error.body,
                });
                this.setGraphBusy(false, '卸载失败', 'error');
                this.toast(error.message, 'error');
            }
        }

        setGraphBusy(busy, text, state = '') {
            this.isGraphBusy = busy;
            this.elements.buildGraphButton.disabled = busy;
            this.elements.openClearModalButton.disabled = busy;
            this.elements.graphOperationState.textContent = text;
            this.elements.graphOperationState.className = 'operation-state';
            if (busy) this.elements.graphOperationState.classList.add('is-working');
            if (state) this.elements.graphOperationState.classList.add(`is-${state}`);
        }

        addActivity(activity) {
            this.activities.unshift({
                id: this.makeId(),
                createdAt: new Date().toISOString(),
                ...activity,
            });
            this.activities = this.activities.slice(0, 20);
            this.writeStorage(ACTIVITY_STORAGE_KEY, this.activities);
            this.renderActivities();
        }

        clearActivities() {
            this.activities = [];
            this.writeStorage(ACTIVITY_STORAGE_KEY, this.activities);
            this.renderActivities();
        }

        renderActivities() {
            if (!Array.isArray(this.activities) || !this.activities.length) {
                this.elements.activityList.innerHTML = '<div class="activity-empty">暂无操作记录</div>';
                return;
            }
            this.elements.activityList.innerHTML = this.activities.map((activity) => {
                const icon = activity.status === 'error'
                    ? 'circle-x'
                    : activity.type === 'clear' ? 'trash-2' : 'database-zap';
                return `
                    <div class="activity-row">
                        <span class="activity-icon is-${this.escapeHtml(activity.status || '')}"><i data-lucide="${icon}" aria-hidden="true"></i></span>
                        <span class="activity-copy">
                            <strong>${this.escapeHtml(activity.title || '操作')}</strong>
                            <span title="${this.escapeHtml(activity.detail || '')}">${this.escapeHtml(activity.detail || '')}</span>
                        </span>
                        <time>${this.escapeHtml(this.formatDateTime(activity.createdAt))}</time>
                    </div>
                `;
            }).join('');
            this.refreshIcons();
        }

        async loadStatus({ silent = false, syncTopK = false } = {}) {
            this.setStatusUi('checking');
            try {
                const status = await this.requestJson('/api/status');
                this.status = status;
                const reachable = Boolean(status.backend?.reachable);
                const runtimeState = status.backend?.status;
                this.setStatusUi(runtimeState === 'loading' ? 'checking' : reachable ? 'online' : 'offline', status);
                const defaultPath = status.defaultCsvPath || 'harrypotter';
                const selectedPath = this.config.csvPath || defaultPath;
                this.elements.csvPathInput.value = selectedPath;
                this.elements.datasetPathLabel.textContent = selectedPath;
                this.elements.apiBaseLabel.textContent = status.apiBase || '未提供';
                this.elements.sidebarStatusAddress.textContent = this.shortAddress(status.apiBase);
                this.elements.timeoutLabel.textContent = `${status.timeoutSeconds ?? 90} 秒`;
                if (reachable && syncTopK) await this.applyTopK({ silent: true });
                if (!silent && !reachable) this.toast(status.backend?.detail || 'LinearRAG 运行时未就绪', 'warning');
            } catch (error) {
                this.setStatusUi('offline');
                if (!silent) this.toast(error.message, 'error');
            }
        }

        setStatusUi(state, status = null) {
            const online = state === 'online';
            const checking = state === 'checking';
            const dotClass = checking ? 'is-checking' : online ? 'is-online' : 'is-offline';
            [this.elements.sidebarStatusDot, this.elements.topbarStatusDot].forEach((dot) => {
                dot.className = `status-dot ${dotClass}`;
            });
            this.elements.sidebarStatusText.textContent = checking ? '正在加载索引' : online ? 'LinearRAG 已就绪' : 'LinearRAG 未加载';
            this.elements.topbarStatusText.textContent = checking ? '加载中' : online ? '索引就绪' : '运行时未加载';
            this.elements.backendInlineState.className = `inline-state ${checking ? 'is-checking' : online ? 'is-online' : 'is-offline'}`;
            this.elements.backendInlineText.textContent = checking
                ? '加载中'
                : online
                    ? `已连接${status?.backend?.latencyMs !== undefined ? ` · ${status.backend.latencyMs} ms` : ''}`
                    : '未加载';
            const welcomeStatus = document.getElementById('welcomeLiveStatus');
            if (welcomeStatus) {
                const dot = welcomeStatus.querySelector('.status-dot');
                if (dot) dot.className = `status-dot ${checking ? 'is-checking' : online ? 'is-online' : 'is-offline'}`;
                const label = welcomeStatus.querySelector('span');
                if (label) label.textContent = checking ? '正在加载 LinearRAG' : online ? 'LinearRAG 已就绪' : '等待加载索引';
            }
        }

        updateSendButton() {
            const hasQuestion = Boolean(this.elements.questionInput.value.trim());
            this.elements.sendButton.disabled = this.isSending || !hasQuestion;
        }

        resizeQuestionInput() {
            const input = this.elements.questionInput;
            input.style.height = 'auto';
            input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
        }

        async requestJson(url, options = {}) {
            let response;
            try {
                response = await fetch(url, {
                    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
                    ...options,
                });
            } catch (error) {
                throw new ApiError('前端代理连接中断', 0, null);
            }

            let body = null;
            try {
                body = await response.json();
            } catch (error) {
                body = { detail: await response.text() };
            }
            if (!response.ok) {
                const message = body?.detail || body?.message || body?.title || `请求失败（HTTP ${response.status}）`;
                throw new ApiError(String(message), response.status, body);
            }
            return body;
        }

        downloadLatestResult() {
            if (!this.latestResult) return;
            const content = JSON.stringify(this.latestResult, null, 2);
            const blob = new Blob([content], { type: 'application/json;charset=utf-8' });
            const url = URL.createObjectURL(blob);
            const anchor = document.createElement('a');
            anchor.href = url;
            anchor.download = `linearrag-result-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
            document.body.appendChild(anchor);
            anchor.click();
            anchor.remove();
            URL.revokeObjectURL(url);
        }

        async copyText(text, successMessage) {
            try {
                await navigator.clipboard.writeText(text);
                this.toast(successMessage, 'success');
            } catch (error) {
                const area = document.createElement('textarea');
                area.value = text;
                document.body.appendChild(area);
                area.select();
                document.execCommand('copy');
                area.remove();
                this.toast(successMessage, 'success');
            }
        }

        toast(message, type = 'success') {
            const toast = document.createElement('div');
            toast.className = `toast is-${type}`;
            const icon = type === 'error' ? 'circle-x' : type === 'warning' ? 'triangle-alert' : 'circle-check';
            toast.innerHTML = `<i data-lucide="${icon}" aria-hidden="true"></i><span>${this.escapeHtml(message)}</span>`;
            this.elements.toastRegion.appendChild(toast);
            this.refreshIcons();
            window.setTimeout(() => {
                toast.classList.add('is-leaving');
                window.setTimeout(() => toast.remove(), 240);
            }, 4200);
        }

        detailEmpty(icon, text) {
            return `<div class="detail-empty"><div class="detail-empty-inner"><i data-lucide="${icon}" aria-hidden="true"></i><span>${this.escapeHtml(text)}</span></div></div>`;
        }

        formatValue(value) {
            if (typeof value === 'string') return value;
            try {
                return JSON.stringify(value, null, 2);
            } catch (error) {
                return String(value);
            }
        }

        formatRelativeTime(value) {
            const timestamp = new Date(value).getTime();
            if (!Number.isFinite(timestamp)) return '';
            const minutes = Math.max(0, Math.floor((Date.now() - timestamp) / 60000));
            if (minutes < 1) return '刚刚';
            if (minutes < 60) return `${minutes} 分钟前`;
            if (minutes < 1440) return `${Math.floor(minutes / 60)} 小时前`;
            return new Date(timestamp).toLocaleDateString('zh-CN');
        }

        formatDateTime(value) {
            const date = new Date(value);
            if (Number.isNaN(date.getTime())) return '';
            return date.toLocaleString('zh-CN', { hour12: false, month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' });
        }

        shortAddress(value) {
            if (!value) return '/api';
            return String(value).replace(/^https?:\/\//, '').replace(/\/api\/v1\/?$/, '');
        }

        persistSessions() {
            this.sessions = this.sessions.slice(0, MAX_SESSIONS);
            this.writeStorage(SESSION_STORAGE_KEY, this.sessions);
        }

        readStorage(key, fallback) {
            try {
                const value = window.localStorage.getItem(key);
                return value ? JSON.parse(value) : fallback;
            } catch (error) {
                return fallback;
            }
        }

        writeStorage(key, value) {
            try {
                window.localStorage.setItem(key, JSON.stringify(value));
            } catch (error) {
                // Storage can be unavailable in private browsing or at quota limits.
            }
        }

        makeId() {
            if (window.crypto?.randomUUID) return window.crypto.randomUUID();
            return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
        }

        escapeHtml(value) {
            return String(value ?? '')
                .replaceAll('&', '&amp;')
                .replaceAll('<', '&lt;')
                .replaceAll('>', '&gt;')
                .replaceAll('"', '&quot;')
                .replaceAll("'", '&#039;');
        }

        refreshIcons() {
            if (window.lucide?.createIcons) {
                window.lucide.createIcons({
                    attrs: {
                        'stroke-width': 1.8,
                        'aria-hidden': 'true',
                    },
                });
            }
        }
    }

    document.addEventListener('DOMContentLoaded', () => {
        window.linearRagWorkbench = new LinearRagWorkbench();
        window.linearRagWorkbench.init();
    });
})();
