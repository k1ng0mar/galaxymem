/**
 * Memory Galaxy — Main app controller (vanilla JS, no React)
 * Coordinates sidebar nav, view switching, search, detail panel.
 */

(function () {
  'use strict';

  // ── Color maps for detail panel ─────────────────────────────────────

  const NETWORK_HEX = {
    world: '#7c8cf8', experience: '#5fb89e', opinion: '#c4a27a', observation: '#9d7cd8',
  };
  const STATUS_HEX = {
    active: '#5fb89e', superseded: '#4a4f5e', contested: '#c4654a',
    demoted: '#3a3e4a', promoted: '#6bc5d8', archived: '#3a3e4a',
  };
  const ENTITY_HEX = {
    person: '#c4a27a', project: '#7c8cf8', self: '#6bc5d8', provisional: '#4a4f5e',
  };

  // ── App ──────────────────────────────────────────────────────────────

  const App = {
    currentView: 'brain',
    networkFilter: '',
    galaxyView: null,
    treeView: null,
    timeTravelMode: false,

    init() {
      this._setupNav();
      this._setupSearch();
      this._setupDetail();
      this._setupModeToggle();
      this._initBrainMap();
    },

    // ── Navigation ────────────────────────────────────────────────────

    _setupNav() {
      document.querySelectorAll('.nav-item[data-view]').forEach(item => {
        item.addEventListener('click', () => {
          const view = item.dataset.view;
          this._switchView(view);
        });
      });
    },

    _switchView(view) {
      this.currentView = view;

      // Update nav highlight
      document.querySelectorAll('.nav-item[data-view]').forEach(item => {
        item.classList.toggle('active', item.dataset.view === view);
      });

      // Show/hide views
      document.querySelectorAll('.view-section').forEach(s => s.classList.remove('active'));
      document.getElementById('view-' + view).classList.add('active');

      // Initialize the view
      if (view === 'brain' && !this.galaxyView) {
        this._initBrainMap();
      } else if (view === 'brain' && this.galaxyView) {
        this.galaxyView._needRedraw = true;
        this.galaxyView._start();
      }

      if (view === 'tree' && !this.treeView) {
        this._initTree();
      }

      if (view === 'stats') {
        this._loadStats();
      }
    },

    // ── Brain map ─────────────────────────────────────────────────────

    _initBrainMap() {
      const canvas = document.getElementById('galaxy-canvas');
      const container = document.getElementById('graph-container');
      if (!canvas || !container) return;

      this.galaxyView = new GalaxyView(canvas, container, this);

      // Zoom buttons
      document.getElementById('zoom-in-btn').addEventListener('click', () => {
        this.galaxyView.zoomBy(1.3);
      });
      document.getElementById('zoom-out-btn').addEventListener('click', () => {
        this.galaxyView.zoomBy(1 / 1.3);
      });

      // Load data
      this.galaxyView.loadData().catch(err => {
        container.innerHTML = `<div class="empty-state"><div class="icon">⚠</div>${err.message}</div>`;
      });
    },

    updateNodeCount(count) {
      document.getElementById('node-count').textContent = count;
    },

    updateFps(fps, alpha) {
      const el = document.getElementById('fps-display');
      if (alpha < 0.001) {
        el.textContent = `${fps} (settled)`;
      } else {
        el.textContent = fps;
      }
    },

    // ── Tree ──────────────────────────────────────────────────────────

    _initTree() {
      const container = document.getElementById('tree-container');
      this.treeView = new TreeView(container, this);
      this.treeView.loadData().catch(err => {
        container.innerHTML = `<div class="empty-state"><div class="icon">⚠</div>${err.message}</div>`;
      });
    },

    // ── Search ────────────────────────────────────────────────────────

    _setupSearch() {
      const input = document.getElementById('search-input');
      let debounce;
      input.addEventListener('input', (e) => {
        clearTimeout(debounce);
        debounce = setTimeout(() => {
          const q = e.target.value;
          if (this.galaxyView) this.galaxyView.applySearch(q);
          if (this.treeView) this.treeView.applySearch(q);
        }, 200);
      });
    },

    // ── Tooltip ───────────────────────────────────────────────────────

    showTooltip(sx, sy, node) {
      const tooltip = document.getElementById('graph-tooltip');
      tooltip.style.display = 'block';
      tooltip.style.left = (sx + 14) + 'px';
      tooltip.style.top = (sy + 14) + 'px';

      if (node.isEntity) {
        const color = ENTITY_HEX[node.entity_type] || '#888';
        tooltip.innerHTML = `<b>${node.label}</b><br><span style="color:${color}">${node.entity_type}</span>`;
      } else {
        const nc = NETWORK_HEX[node.network] || '#888';
        const sc = STATUS_HEX[node.status] || '#888';
        const text = node.label ? node.label.substring(0, 100) : '';
        tooltip.innerHTML = `<span style="color:${nc}">${node.network}</span> <span style="color:${sc}">${node.status}</span><br>${text}`;
      }
    },

    hideTooltip() {
      document.getElementById('graph-tooltip').style.display = 'none';
    },

    // ── Detail panel ──────────────────────────────────────────────────

    _setupDetail() {
      document.getElementById('detail-close').addEventListener('click', () => {
        this._closeDetail();
      });
    },

    async onSelectNode(node) {
      if (!node) return;
      const panel = document.getElementById('detail-panel');
      const content = document.getElementById('detail-content');
      panel.classList.add('open');
      content.innerHTML = '<div class="loading">Loading</div>';

      try {
        let detail;
        if (node.isEntity) {
          const res = await fetch(`/entity/${node.id}`);
          if (!res.ok) throw new Error('Not found');
          detail = await res.json();
          this._renderEntityDetail(detail);
        } else {
          const res = await fetch(`/memory/${node.id}`);
          if (!res.ok) throw new Error('Not found');
          detail = await res.json();
          this._renderMemoryDetail(detail);
        }
      } catch (err) {
        content.innerHTML = `<div class="empty-state"><div class="icon">⚠</div>${err.message}</div>`;
      }
    },

    _closeDetail() {
      document.getElementById('detail-panel').classList.remove('open');
      if (this.galaxyView) {
        this.galaxyView.selectNode(null);
      }
    },

    _renderMemoryDetail(detail) {
      const m = detail.memory;
      const content = document.getElementById('detail-content');

      let html = '';

      html += `<div class="detail-section">
        <div class="detail-label">Network</div>
        <div class="detail-value"><span class="stat-marker" style="background:${NETWORK_HEX[m.network] || '#888'}"></span>${m.network}</div>
      </div>`;

      html += `<div class="detail-section">
        <div class="detail-label">Status</div>
        <div class="detail-value" style="color:${STATUS_HEX[m.status] || '#888'}">${m.status}</div>
      </div>`;

      if (m.brightness !== undefined) {
        html += `<div class="detail-section">
          <div class="detail-label">Brightness</div>
          <div class="detail-value" style="display:flex;align-items:center;gap:8px">
            <div class="brightness-bar"><div class="brightness-fill" style="width:${Math.round(m.brightness * 100)}%"></div></div>
            ${m.brightness.toFixed(3)}
          </div>
        </div>`;
      }

      html += `<div class="detail-section">
        <div class="detail-label">Text</div>
        <div class="detail-text">${this._esc(m.text)}</div>
      </div>`;

      if (m.created_at) {
        html += `<div class="detail-section">
          <div class="detail-label">Created</div>
          <div class="detail-value">${new Date(m.created_at).toLocaleString()}</div>
        </div>`;
      }

      if (m.last_recalled_at) {
        html += `<div class="detail-section">
          <div class="detail-label">Last Recalled</div>
          <div class="detail-value">${new Date(m.last_recalled_at).toLocaleString()}</div>
        </div>`;
      }

      html += `<div class="detail-section">
        <div class="detail-label">Recall Count</div>
        <div class="detail-value">${m.recall_count || 0}</div>
      </div>`;

      if (m.entity_ids && m.entity_ids.length > 0) {
        html += `<div class="detail-section">
          <div class="detail-label">Entities</div>
          <div class="detail-value">${m.entity_ids.join(', ')}</div>
        </div>`;
      }

      if (detail.edges && detail.edges.length > 0) {
        html += `<div class="detail-section">
          <div class="detail-label">Edges (${detail.edges.length})</div>
          <div class="edge-list">`;
        for (const e of detail.edges.slice(0, 20)) {
          const dir = e.from_id === m.id ? '→' : '←';
          const targetId = e.from_id === m.id ? e.to_id : e.from_id;
          html += `<div class="edge-item">
            <span class="filter-dot" style="background:${STATUS_HEX[e.kind] || '#4a4f5e'}"></span>
            <span class="edge-kind">${e.kind}</span>
            <span style="color:var(--text-muted)">${dir} ${targetId.substring(0, 8)}</span>
          </div>`;
        }
        html += `</div></div>`;
      }

      content.innerHTML = html;
    },

    _renderEntityDetail(detail) {
      const e = detail.entity;
      const content = document.getElementById('detail-content');

      let html = '';

      html += `<div class="detail-section">
        <div class="detail-label">Type</div>
        <div class="detail-value"><span class="stat-marker" style="background:${ENTITY_HEX[e.type] || '#888'}"></span>${e.type}</div>
      </div>`;

      html += `<div class="detail-section">
        <div class="detail-label">Label</div>
        <div class="detail-text">${this._esc(e.label)}</div>
      </div>`;

      if (e.status_line) {
        html += `<div class="detail-section">
          <div class="detail-label">Status Line</div>
          <div class="detail-value">${this._esc(e.status_line)}</div>
        </div>`;
      }

      if (e.card && Object.keys(e.card).length > 0) {
        html += `<div class="detail-section">
          <div class="detail-label">Card</div>
          <div class="detail-text">`;
        for (const [k, v] of Object.entries(e.card)) {
          html += `<div><strong>${k}:</strong> ${v}</div>`;
        }
        html += `</div></div>`;
      }

      if (detail.memories && detail.memories.length > 0) {
        html += `<div class="detail-section">
          <div class="detail-label">Memories (${detail.memories.length})</div>
          <div class="edge-list">`;
        for (const m of detail.memories.slice(0, 20)) {
          const text = m.text && m.text.length > 60 ? m.text.substring(0, 60) + '...' : m.text;
          html += `<div class="edge-item" style="cursor:pointer" data-mem-id="${m.id}">
            <span class="filter-dot" style="background:${NETWORK_HEX[m.network] || '#888'}"></span>
            <span style="color:var(--text-secondary);overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${this._esc(text)}</span>
          </div>`;
        }
        html += `</div></div>`;

        // Wire up clickable memories after render
        setTimeout(() => {
          content.querySelectorAll('[data-mem-id]').forEach(el => {
            el.addEventListener('click', () => {
              this.onSelectNode({ id: el.dataset.memId, isEntity: false });
            });
          });
        }, 0);
      }

      content.innerHTML = html;
    },

    // ── Stats ─────────────────────────────────────────────────────────

    async _loadStats() {
      const container = document.getElementById('stats-container');
      container.innerHTML = '<div class="loading">Fetching stats</div>';

      try {
        const res = await fetch('/api/stats');
        const stats = await res.json();
        this._renderStats(stats);
      } catch (err) {
        container.innerHTML = `<div class="empty-state"><div class="icon">⚠</div>${err.message}</div>`;
      }
    },

    _renderStats(stats) {
      const container = document.getElementById('stats-container');
      const total = stats.total_memories || 0;
      const netData = stats.memories_per_network || {};
      const statusData = stats.memories_per_status || {};
      const entData = stats.entities_per_type || {};

      const networks = [
        { key: 'world', label: 'World', color: '#7c8cf8' },
        { key: 'experience', label: 'Experience', color: '#5fb89e' },
        { key: 'opinion', label: 'Opinion', color: '#c4a27a' },
        { key: 'observation', label: 'Observation', color: '#9d7cd8' },
      ];

      const statuses = [
        { key: 'active', label: 'Active', color: '#5fb89e' },
        { key: 'superseded', label: 'Superseded', color: '#4a4f5e' },
        { key: 'contested', label: 'Contested', color: '#c4654a' },
        { key: 'demoted', label: 'Demoted', color: '#3a3e4a' },
        { key: 'archived', label: 'Archived', color: '#3a3e4a' },
      ];

      const entities = [
        { key: 'self', label: 'Self', color: '#6bc5d8' },
        { key: 'person', label: 'Person', color: '#c4a27a' },
        { key: 'project', label: 'Project', color: '#7c8cf8' },
        { key: 'provisional', label: 'Provisional', color: '#4a4f5e' },
      ];

      let html = '<div class="page-title">Statistics</div>';

      // Top stats
      html += '<div class="stats-grid">';
      html += `<div class="stat-card"><div class="stat-label">Memories</div><div class="stat-value">${total.toLocaleString()}</div></div>`;
      html += `<div class="stat-card"><div class="stat-label">Entities</div><div class="stat-value">${stats.total_entities || 0}</div></div>`;
      html += `<div class="stat-card"><div class="stat-label">Edges</div><div class="stat-value">${stats.total_edges || 0}</div></div>`;
      html += `<div class="stat-card"><div class="stat-label">Unprocessed</div><div class="stat-value">${stats.unprocessed_flags || 0}</div></div>`;
      html += '</div>';

      // By network
      html += '<div class="section-title">By Network</div><div class="stats-grid">';
      for (const n of networks) {
        const count = netData[n.key] || 0;
        const pct = total > 0 ? (count / total * 100).toFixed(1) : 0;
        html += `<div class="stat-card"><div class="stat-label"><span class="stat-marker" style="background:${n.color}"></span>${n.label}</div><div class="stat-value">${count.toLocaleString()}</div><div class="stat-detail">${pct}%</div></div>`;
      }
      html += '</div>';

      // By status
      html += '<div class="section-title" style="margin-top:24px">By Status</div><div class="stats-grid">';
      for (const s of statuses) {
        const count = statusData[s.key] || 0;
        html += `<div class="stat-card"><div class="stat-label"><span class="stat-marker" style="background:${s.color}"></span>${s.label}</div><div class="stat-value">${count.toLocaleString()}</div></div>`;
      }
      html += '</div>';

      // By entity type
      html += '<div class="section-title" style="margin-top:24px">By Entity Type</div><div class="stats-grid">';
      for (const e of entities) {
        const count = entData[e.key] || 0;
        html += `<div class="stat-card"><div class="stat-label"><span class="stat-marker" style="background:${e.color}"></span>${e.label}</div><div class="stat-value">${count.toLocaleString()}</div></div>`;
      }
      html += '</div>';

      container.innerHTML = html;
    },

    // ── Mode toggle (current vs time-travel) ──────────────────────────

    _setupModeToggle() {
      const currentBtn = document.getElementById('mode-current');
      const travelBtn = document.getElementById('mode-timetravel');

      currentBtn.addEventListener('click', () => {
        this.timeTravelMode = false;
        currentBtn.classList.add('active');
        travelBtn.classList.remove('active');
        if (this.galaxyView) {
          this.galaxyView.loadData();
        }
      });

      travelBtn.addEventListener('click', () => {
        // Simple prompt for timestamp — could be a date picker later
        const ts = prompt('Enter ISO timestamp (e.g. 2025-06-01T00:00:00):');
        if (!ts) return;
        this.timeTravelMode = true;
        travelBtn.classList.add('active');
        currentBtn.classList.remove('active');
        if (this.galaxyView) {
          this.galaxyView.loadData(ts);
        }
      });
    },

    // ── Utils ─────────────────────────────────────────────────────────

    _esc(str) {
      if (!str) return '';
      const div = document.createElement('div');
      div.textContent = str;
      return div.innerHTML;
    },
  };

  // Boot — handle both cases: script at end of <body> means DOM is already
  // parsed (readyState !== 'loading'), so DOMContentLoaded already fired and a
  // late listener would never run. Call init() immediately in that case.
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => App.init());
  } else {
    App.init();
  }

  window.App = App;
})();
