/**
 * Memory Galaxy — Tree View
 * Hierarchical: entity → network → memories.
 * Pure DOM (expandable tree), no per-node canvas. Uses same /graph data.
 */

(function () {
  'use strict';

  const NETWORK_RGB = {
    world: [124, 140, 248],
    experience: [95, 184, 158],
    opinion: [196, 162, 122],
    observation: [157, 124, 216],
  };

  const ENTITY_RGB = {
    person: [196, 162, 122],
    project: [124, 140, 248],
    self: [107, 197, 216],
    provisional: [74, 79, 94],
  };

  const STATUS_RGB = {
    active: [95, 184, 158],
    superseded: [74, 79, 94],
    contested: [196, 101, 74],
    demoted: [58, 62, 74],
    promoted: [107, 197, 216],
    archived: [58, 62, 74],
  };

  function rgba(rgb, a) {
    return `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${a})`;
  }

  class TreeView {
    constructor(container, app) {
      this.container = container;
      this.app = app;
      this.data = null;
      this.expanded = new Set();
      this.searchQuery = '';
    }

    async loadData() {
      const res = await fetch('/graph?limit=500');
      if (!res.ok) throw new Error(`Graph API error: ${res.status}`);
      this.data = await res.json();
      this._render();
    }

    _buildHierarchy() {
      // Group memories by entity → network
      const entityMap = {};
      for (const e of this.data.entities || []) {
        entityMap[e.id] = {
          entity: e,
          networks: {},
          unscoped: [],
        };
      }

      for (const m of this.data.memories || []) {
        if (m.status === 'superseded') continue; // hidden in normal mode

        if (!m.entity_ids || m.entity_ids.length === 0) {
          // Unscoped — goes under a virtual "general" node
          if (!entityMap['__general__']) {
            entityMap['__general__'] = {
              entity: { id: '__general__', label: 'Unscoped', entity_type: 'provisional', status_line: '' },
              networks: {},
              unscoped: [],
            };
          }
          const net = m.network || 'world';
          if (!entityMap['__general__'].networks[net]) {
            entityMap['__general__'].networks[net] = [];
          }
          entityMap['__general__'].networks[net].push(m);
          continue;
        }

        for (const eid of m.entity_ids) {
          if (!entityMap[eid]) continue;
          const net = m.network || 'world';
          if (!entityMap[eid].networks[net]) {
            entityMap[eid].networks[net] = [];
          }
          entityMap[eid].networks[net].push(m);
        }
      }

      return Object.values(entityMap).sort((a, b) => {
        // self first, then by memory count desc
        if (a.entity.entity_type === 'self') return -1;
        if (b.entity.entity_type === 'self') return 1;
        const ac = Object.values(a.networks).flat().length;
        const bc = Object.values(b.networks).flat().length;
        return bc - ac;
      });
    }

    _render() {
      const hierarchy = this._buildHierarchy();
      const q = this.searchQuery.toLowerCase();

      const container = this.container;
      container.innerHTML = '';

      const tree = document.createElement('div');
      tree.className = 'tree-root';

      for (const entGroup of hierarchy) {
        const e = entGroup.entity;
        const allMems = Object.values(entGroup.networks).flat();
        if (allMems.length === 0 && e.id !== '__general__') {
          // Show entity even with 0 memories
        }

        // Filter by search
        let filteredMems = allMems;
        if (q) {
          filteredMems = allMems.filter(m =>
            (m.text || '').toLowerCase().includes(q) ||
            (m.label || '').toLowerCase().includes(q)
          );
          if (filteredMems.length === 0) continue; // skip entity with no matches
        }

        const entNode = document.createElement('div');
        entNode.className = 'tree-entity';

        const entRow = document.createElement('div');
        entRow.className = 'tree-row tree-entity-row';

        const expandIcon = document.createElement('span');
        expandIcon.className = 'tree-expand';
        expandIcon.textContent = this.expanded.has(e.id) ? '▾' : '▸';
        entRow.appendChild(expandIcon);

        const dot = document.createElement('span');
        dot.className = 'tree-dot';
        const ergb = ENTITY_RGB[e.entity_type] || [130, 130, 140];
        dot.style.background = rgba(ergb, 0.8);
        if (e.entity_type === 'person' || e.entity_type === 'self') {
          dot.classList.add('dashed');
        }
        entRow.appendChild(dot);

        const label = document.createElement('span');
        label.className = 'tree-label';
        label.textContent = e.label;
        entRow.appendChild(label);

        const count = document.createElement('span');
        count.className = 'tree-count';
        count.textContent = allMems.length;
        entRow.appendChild(count);

        if (e.status_line) {
          const status = document.createElement('span');
          status.className = 'tree-status-line';
          status.textContent = e.status_line;
          entRow.appendChild(status);
        }

        entNode.appendChild(entRow);

        // Expand/collapse
        const childContainer = document.createElement('div');
        childContainer.className = 'tree-children';
        childContainer.style.display = this.expanded.has(e.id) ? 'block' : 'none';

        entRow.addEventListener('click', (ev) => {
          ev.stopPropagation();
          if (this.expanded.has(e.id)) {
            this.expanded.delete(e.id);
            childContainer.style.display = 'none';
            expandIcon.textContent = '▸';
          } else {
            this.expanded.add(e.id);
            childContainer.style.display = 'block';
            expandIcon.textContent = '▾';
          }
        });

        // Click entity label → open detail
        label.style.cursor = 'pointer';
        label.addEventListener('click', (ev) => {
          ev.stopPropagation();
          if (e.id !== '__general__') {
            this.app.onSelectNode({ id: e.id, isEntity: true });
          }
        });

        // Build network groups
        for (const [net, mems] of Object.entries(entGroup.networks)) {
          let netMems = mems;
          if (q) {
            netMems = mems.filter(m =>
              (m.text || '').toLowerCase().includes(q) ||
              (m.label || '').toLowerCase().includes(q)
            );
            if (netMems.length === 0) continue;
          }

          const netRow = document.createElement('div');
          netRow.className = 'tree-row tree-network-row';

          const netExpand = document.createElement('span');
          netExpand.className = 'tree-expand';
          const netKey = e.id + '::' + net;
          netExpand.textContent = this.expanded.has(netKey) ? '▾' : '▸';
          netRow.appendChild(netExpand);

          const netDot = document.createElement('span');
          netDot.className = 'tree-dot';
          const nrgb = NETWORK_RGB[net] || [130, 130, 140];
          netDot.style.background = rgba(nrgb, 0.8);
          netRow.appendChild(netDot);

          const netLabel = document.createElement('span');
          netLabel.className = 'tree-label tree-network-label';
          netLabel.textContent = net;
          netRow.appendChild(netLabel);

          const netCount = document.createElement('span');
          netCount.className = 'tree-count';
          netCount.textContent = netMems.length;
          netRow.appendChild(netCount);

          childContainer.appendChild(netRow);

          const memContainer = document.createElement('div');
          memContainer.className = 'tree-children tree-mem-container';
          memContainer.style.display = this.expanded.has(netKey) ? 'block' : 'none';

          netRow.addEventListener('click', (ev) => {
            ev.stopPropagation();
            if (this.expanded.has(netKey)) {
              this.expanded.delete(netKey);
              memContainer.style.display = 'none';
              netExpand.textContent = '▸';
            } else {
              this.expanded.add(netKey);
              memContainer.style.display = 'block';
              netExpand.textContent = '▾';
            }
          });

          // Auto-expand first level if search is active
          if (q && !this.expanded.has(e.id)) {
            this.expanded.add(e.id);
            childContainer.style.display = 'block';
            expandIcon.textContent = '▾';
            this.expanded.add(netKey);
            memContainer.style.display = 'block';
            netExpand.textContent = '▾';
          }

          // Memory leaves
          for (const m of netMems) {
            const memRow = document.createElement('div');
            memRow.className = 'tree-row tree-memory-row';

            // Status indicator
            const srgb = STATUS_RGB[m.status] || [130, 130, 140];
            const statusDot = document.createElement('span');
            statusDot.className = 'tree-dot tree-status-dot';
            statusDot.style.background = rgba(srgb, 0.7);
            memRow.appendChild(statusDot);

            const memText = document.createElement('span');
            memText.className = 'tree-memory-text';
            const brightness = m.brightness !== undefined ? m.brightness : 1.0;
            memText.style.opacity = Math.max(0.3, brightness);
            memText.textContent = m.text && m.text.length > 80 ? m.text.substring(0, 80) + '...' : (m.text || m.label || '');
            memRow.appendChild(memText);

            const rc = document.createElement('span');
            rc.className = 'tree-count';
            rc.textContent = `rc:${m.recall_count || 0}`;
            memRow.appendChild(rc);

            // Contested marker
            if (m.status === 'contested') {
              memRow.classList.add('contested');
            }
            if (m.status === 'demoted') {
              memRow.classList.add('demoted');
            }

            memRow.addEventListener('click', (ev) => {
              ev.stopPropagation();
              this.app.onSelectNode({ id: m.id, isEntity: false });
            });

            memContainer.appendChild(memRow);
          }

          childContainer.appendChild(memContainer);
        }

        entNode.appendChild(childContainer);
        tree.appendChild(entNode);
      }

      if (tree.children.length === 0) {
        container.innerHTML = '<div class="empty-state"><div class="icon">∅</div>No memories found</div>';
        return;
      }

      container.appendChild(tree);
    }

    applySearch(query) {
      this.searchQuery = query || '';
      if (this.data) this._render();
    }

    destroy() {
      this.container.innerHTML = '';
    }
  }

  window.TreeView = TreeView;
})();
