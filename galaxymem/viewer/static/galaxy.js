/**
 * Memory Galaxy — Canvas renderer
 * Single <canvas>, d3-force for physics (decoupled), alpha-cooldown stop,
 * no shadowBlur, DPR capped at 2 (1.5 touch), off-screen culling.
 *
 * Brightness (recall decay) = fill-alpha, computed once per data update.
 * Person nodes: dashed outline. Project nodes: solid outline.
 * Contested: split/flicker marker. Superseded: hidden (except time-travel).
 * Demoted: dimmed below normal decay floor.
 */

(function () {
  'use strict';

  // ── Color palette (from prototype, unchanged) ────────────────────────

  const NETWORK_RGB = {
    world: [124, 140, 248],
    experience: [95, 184, 158],
    opinion: [196, 162, 122],
    observation: [157, 124, 216],
  };

  const STATUS_RGB = {
    active: [95, 184, 158],
    superseded: [74, 79, 94],
    contested: [196, 101, 74],
    demoted: [58, 62, 74],
    promoted: [107, 197, 216],
    archived: [58, 62, 74],
  };

  const ENTITY_RGB = {
    person: [196, 162, 122],
    project: [124, 140, 248],
    self: [107, 197, 216],
    provisional: [74, 79, 94],
  };

  const EDGE_RGB = {
    shared_entity: [120, 200, 175],
    temporal: [140, 155, 255],
    derived_from: [210, 175, 135],
    supersedes: [140, 145, 160],
    contests: [210, 115, 85],
  };

  function rgba(rgb, alpha) {
    return `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${alpha})`;
  }

  // ── Starfield (subtle background, same aesthetic as prototype) ───────

  class Starfield {
    constructor() {
      this.stars = [];
      this.nebulas = [];
    }

    generate(w, h) {
      const count = Math.floor((w * h) / 6000);
      this.stars = [];
      for (let i = 0; i < count; i++) {
        this.stars.push({
          x: Math.random() * w,
          y: Math.random() * h,
          r: Math.random() * 0.8 + 0.15,
          op: Math.random() * 0.2 + 0.04,
          tp: Math.random() * Math.PI * 2,
          ts: Math.random() * 0.0008 + 0.0002,
        });
      }

      const nebulaColors = [[60, 50, 100], [40, 55, 90], [50, 40, 80]];
      this.nebulas = [];
      for (let i = 0; i < 3; i++) {
        this.nebulas.push({
          x: Math.random() * w,
          y: Math.random() * h,
          r: 200 + Math.random() * 250,
          c: nebulaColors[i],
          op: 0.025 + Math.random() * 0.03,
        });
      }
    }

    draw(ctx, w, h, time, panX, panY) {
      // Nebulas (extremely faint)
      for (const n of this.nebulas) {
        const px = n.x + panX * 0.08;
        const py = n.y + panY * 0.08;
        const grad = ctx.createRadialGradient(px, py, 0, px, py, n.r);
        grad.addColorStop(0, rgba(n.c, n.op));
        grad.addColorStop(1, rgba(n.c, 0));
        ctx.fillStyle = grad;
        ctx.fillRect(px - n.r, py - n.r, n.r * 2, n.r * 2);
      }

      // Stars (tiny, cold)
      for (const s of this.stars) {
        const px = ((s.x + panX * 0.15) % w + w) % w;
        const py = ((s.y + panY * 0.15) % h + h) % h;
        const twinkle = Math.sin(time * s.ts + s.tp) * 0.4 + 0.6;
        ctx.fillStyle = `rgba(180,195,240,${s.op * twinkle})`;
        ctx.beginPath();
        ctx.arc(px, py, s.r, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  // ── Galaxy View ──────────────────────────────────────────────────────

  class GalaxyView {
    constructor(canvas, container, app) {
      this.canvas = canvas;
      this.container = container;
      this.app = app;
      this.ctx = canvas.getContext('2d', { alpha: false });
      this.starfield = new Starfield();

      // ── Gradient cache: Map<"r,g,b,alphaBucket" → RadialGradient> ──
      // Prevents createRadialGradient from being called per-node per-frame.
      this._gradCache = new Map();

      // State
      this.nodes = [];
      this.nodeMap = {};
      this.edges = [];
      this.entityNodes = [];
      this.memoryNodes = [];
      this.selectedNode = null;
      this.hoverNode = null;
      this.activatedSet = null; // spreading activation neighborhood

      // Camera
      this.panX = 0;
      this.panY = 0;
      this.zoom = 1;
      this.targetZoom = 1;
      this.targetPanX = 0;
      this.targetPanY = 0;

      // Interaction
      this.isDragging = false;
      this.dragNode = null;
      this.isPanning = false;
      this.dragStartX = 0;
      this.dragStartY = 0;
      this.lastX = 0;
      this.lastY = 0;

      // Simulation (decoupled from rendering)
      this.alpha = 1;
      this.alphaDecay = 0.0228; // d3-force default ~0.0228
      this.alphaMin = 0.001; // stop threshold
      this.alphaTarget = 0;
      this.isRunning = false;

      // Search
      this.searchQuery = '';
      this.searchDims = new Set(); // non-matching node ids to dim

      // FPS tracking
      this.frameCount = 0;
      this.lastFpsTime = performance.now();
      this.fps = 0;

      // Bind
      this._tick = this._tick.bind(this);
      this._onMouseDown = this._onMouseDown.bind(this);
      this._onMouseMove = this._onMouseMove.bind(this);
      this._onMouseUp = this._onMouseUp.bind(this);
      this._onMouseLeave = this._onMouseLeave.bind(this);
      this._onWheel = this._onWheel.bind(this);
      this._onTouchStart = this._onTouchStart.bind(this);
      this._onTouchMove = this._onTouchMove.bind(this);
      this._onTouchEnd = this._onTouchEnd.bind(this);
      this._onResize = this._onResize.bind(this);

      this._setupCanvas();
      this._setupEvents();
    }

    // ── Canvas setup ──────────────────────────────────────────────────

    _dpr() {
      const isTouch = 'ontouchstart' in window;
      return Math.min(window.devicePixelRatio || 1, isTouch ? 1.5 : 2);
    }

    _setupCanvas() {
      this._resize();
    }

    _resize() {
      const w = this.container.clientWidth;
      const h = this.container.clientHeight;
      const dpr = this._dpr();
      this.canvas.width = w * dpr;
      this.canvas.height = h * dpr;
      this.canvas.style.width = w + 'px';
      this.canvas.style.height = h + 'px';
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.starfield.generate(w, h);
    }

    _onResize() {
      this._resize();
      this._draw(performance.now());
    }

    _W() { return this.container.clientWidth; }
    _H() { return this.container.clientHeight; }

    // ── Event setup ───────────────────────────────────────────────────

    _setupEvents() {
      this.canvas.addEventListener('mousedown', this._onMouseDown);
      this.canvas.addEventListener('mousemove', this._onMouseMove);
      this.canvas.addEventListener('mouseup', this._onMouseUp);
      this.canvas.addEventListener('mouseleave', this._onMouseLeave);
      this.canvas.addEventListener('wheel', this._onWheel, { passive: false });
      this.canvas.addEventListener('touchstart', this._onTouchStart, { passive: false });
      this.canvas.addEventListener('touchmove', this._onTouchMove, { passive: false });
      this.canvas.addEventListener('touchend', this._onTouchEnd);
      window.addEventListener('resize', this._onResize);
    }

    destroy() {
      this.canvas.removeEventListener('mousedown', this._onMouseDown);
      this.canvas.removeEventListener('mousemove', this._onMouseMove);
      this.canvas.removeEventListener('mouseup', this._onMouseUp);
      this.canvas.removeEventListener('mouseleave', this._onMouseLeave);
      this.canvas.removeEventListener('wheel', this._onWheel);
      this.canvas.removeEventListener('touchstart', this._onTouchStart);
      this.canvas.removeEventListener('touchmove', this._onTouchMove);
      this.canvas.removeEventListener('touchend', this._onTouchEnd);
      window.removeEventListener('resize', this._onResize);
      this.isRunning = false;
    }

    // ── Load data from /graph ─────────────────────────────────────────

    async loadData(timeTravel) {
      let url = '/graph';
      if (timeTravel) {
        url = '/as-of/' + encodeURIComponent(timeTravel);
      }
      const res = await fetch(url);
      if (!res.ok) throw new Error(`Graph API error: ${res.status}`);
      const data = await res.json();

      this._buildGraph(data);
      this.alpha = 1;
      this._start();
    }

    _buildGraph(data) {
      this.brightnessFloor = data.brightness_floor || 0.15;
      this.isTimeTravel = !!data.as_of;
      this._fitted = false;

      // Build entity nodes
      this.entityNodes = (data.entities || []).map((e, i) => ({
        ...e,
        x: this._W() / 2 + Math.cos(i / Math.max(data.entities.length, 1) * Math.PI * 2) * (120 + Math.random() * 60),
        y: this._H() / 2 + Math.sin(i / Math.max(data.entities.length, 1) * Math.PI * 2) * (120 + Math.random() * 60),
        vx: 0, vy: 0,
        radius: 7,
        isEntity: true,
        pulsePhase: Math.random() * Math.PI * 2,
        fillAlpha: 1.0,
      }));

      this.entityMap = {};
      this.entityNodes.forEach(n => { this.entityMap[n.id] = n; });

      // Build memory nodes — skip superseded unless time-travel mode
      this.memoryNodes = (data.memories || [])
        .filter(m => this.isTimeTravel || m.status !== 'superseded')
        .map(m => {
          // Brightness is precomputed by backend; we just use it as fill-alpha
          let fillAlpha = m.brightness !== undefined ? m.brightness : 1.0;

          // Demoted: dim below decay floor
          if (m.status === 'demoted') {
            fillAlpha = Math.min(fillAlpha, this.brightnessFloor * 0.4);
          }

          return {
            ...m,
            x: this._W() / 2 + (Math.random() - 0.5) * 300,
            y: this._H() / 2 + (Math.random() - 0.5) * 300,
            vx: 0, vy: 0,
            radius: Math.max(1.5, Math.min(4, 1.5 + (m.recall_count || 0) * 0.15)),
            isEntity: false,
            fillAlpha: fillAlpha,
            flickerPhase: Math.random() * Math.PI * 2,
            // Precompute static fill style (brightness+network fixed per node)
            style: rgba(NETWORK_RGB[m.network] || [130, 130, 140], fillAlpha),
          };
        });

      this.nodes = [...this.entityNodes, ...this.memoryNodes];
      this.nodeMap = {};
      this.nodes.forEach(n => { this.nodeMap[n.id] = n; });

      // Build edges with resolved node refs
      this.edges = (data.edges || [])
        .filter(e => this.nodeMap[e.source] && this.nodeMap[e.target])
        .map(e => ({
          ...e,
          s: this.nodeMap[e.source],
          t: this.nodeMap[e.target],
        }));

      // Update FPS/node info display
      if (this.app) {
        this.app.updateNodeCount(this.nodes.length);
      }
    }

    // ── Simulation (d3-force style, decoupled from rendering) ─────────

    // ── Barnes-Hut Quadtree for O(n log n) repulsion ─────────────────
    // Quadtree node: { data, nw, ne, se, sw, x0, y0, x1, y1 }
    _bhBuild(nodes) {
      const w = this._W(), h = this._H();
      let qt = { data: null, nw: null, ne: null, se: null, sw: null, x0: 0, y0: 0, x1: w, y1: h };
      for (const d of nodes) {
        qt = this._bhInsert(qt, d);
      }
      return qt;
    }

    _bhInsert(q, d) {
      if (q.data === null) { q.data = d; return q; }
      if (q.nw === null) {
        // Split
        const mx = (q.x0 + q.x1) / 2, my = (q.y0 + q.y1) / 2;
        q.nw = { data: null, nw: null, ne: null, se: null, sw: null, x0: q.x0, y0: q.y0, x1: mx, y1: my };
        q.ne = { data: null, nw: null, ne: null, se: null, sw: null, x0: mx, y0: q.y0, x1: q.x1, y1: my };
        q.se = { data: null, nw: null, ne: null, se: null, sw: null, x0: mx, y0: my, x1: q.x1, y1: q.y1 };
        q.sw = { data: null, nw: null, ne: null, se: null, sw: null, x0: q.x0, y0: my, x1: mx, y1: q.y1 };
        // Re-insert existing data
        if (q.nw.data) q = this._bhInsert(q, q.nw.data); q.nw.data = null;
        if (q.ne.data) q = this._bhInsert(q, q.ne.data); q.ne.data = null;
        if (q.se.data) q = this._bhInsert(q, q.se.data); q.se.data = null;
        if (q.sw.data) q = this._bhInsert(q, q.sw.data); q.sw.data = null;
      }
      const mx = (q.x0 + q.x1) / 2, my = (q.y0 + q.y1) / 2;
      if (d.x < mx && d.y < my) return this._bhInsert(q.nw, d);
      if (d.x >= mx && d.y < my) return this._bhInsert(q.ne, d);
      if (d.x >= mx && d.y >= my) return this._bhInsert(q.se, d);
      return this._bhInsert(q.sw, d);
    }

    _bhForce(qt, source, strength, result) {
      this._bhTraverse(qt, source, strength, result, 0.7);
    }

    _bhTraverse(q, source, strength, result, theta) {
      if (q.data !== null && q.data === source) return; // skip self
      if (q.nw === null) {
        // Leaf node with data
        if (q.data === null) return;
        const sx = q.data.x, sy = q.data.y;
        let ddx = sx - source.x, ddy = sy - source.y;
        let d2 = ddx * ddx + ddy * ddy;
        if (d2 < 1) d2 = 1;
        const d = Math.sqrt(d2);
        const f = strength / d2;
        result.vx += (ddx / d) * f;
        result.vy += (ddy / d) * f;
        return;
      }
      // Internal node — Barnes-Hut test
      const cx = (q.x0 + q.x1) / 2, cy = (q.y0 + q.y1) / 2;
      const size = q.x1 - q.x0;
      let ddx = cx - source.x, ddy = cy - source.y;
      let d2 = ddx * ddx + ddy * ddy;
      if (d2 < 1) d2 = 1;
      const d = Math.sqrt(d2);
      if (size / d < theta) {
        // Approximate: count nodes in subtree
        let count = this._bhCount(q);
        const f = strength * count / d2;
        result.vx += (ddx / d) * f;
        result.vy += (ddy / d) * f;
      } else {
        if (q.nw) this._bhTraverse(q.nw, source, strength, result, theta);
        if (q.ne) this._bhTraverse(q.ne, source, strength, result, theta);
        if (q.se) this._bhTraverse(q.se, source, strength, result, theta);
        if (q.sw) this._bhTraverse(q.sw, source, strength, result, theta);
      }
    }

    _bhCount(q) {
      if (q.nw === null) return q.data !== null ? 1 : 0;
      return (q.nw ? this._bhCount(q.nw) : 0) +
             (q.ne ? this._bhCount(q.ne) : 0) +
             (q.se ? this._bhCount(q.se) : 0) +
             (q.sw ? this._bhCount(q.sw) : 0);
    }

    _start() {
      if (this.isRunning) return;
      this.isRunning = true;
      this.alpha = Math.max(this.alpha, 0.1);
      requestAnimationFrame(this._tick);
    }

    _restartAlpha(alpha = 0.3) {
      this.alpha = Math.max(this.alpha, alpha);
      if (!this.isRunning) {
        this.isRunning = true;
        requestAnimationFrame(this._tick);
      }
    }

    _tick(time) {
      // FPS
      this.frameCount++;
      if (time - this.lastFpsTime >= 1000) {
        this.fps = Math.round(this.frameCount * 1000 / (time - this.lastFpsTime));
        this.frameCount = 0;
        this.lastFpsTime = time;
        if (this.app) this.app.updateFps(this.fps, this.alpha);
      }

      // Camera interpolation
      this.zoom += (this.targetZoom - this.zoom) * 0.15;
      this.panX += (this.targetPanX - this.panX) * 0.15;
      this.panY += (this.targetPanY - this.panY) * 0.15;
      const cameraMoved = Math.abs(this.targetZoom - this.zoom) > 0.001 ||
                          Math.abs(this.targetPanX - this.panX) > 0.5 ||
                          Math.abs(this.targetPanY - this.panY) > 0.5;

      // ── Alpha cooldown: stop the simulation when alpha < alphaMin ──
      if (this.alpha > this.alphaMin) {
        this.alpha = Math.max(this.alphaMin, this.alpha - this.alphaDecay);
        this._applyForces(this.alpha);
        this._draw(time);
        requestAnimationFrame(this._tick);
      } else {
        // Sim settled — fit camera to actual node bounding box once, then idle
        if (!this._fitted) {
          this._fitToBounds();
          this._fitted = true;
        }
        if (cameraMoved || this.hoverNode || this._needRedraw) {
          this._draw(time);
          this._needRedraw = false;
          requestAnimationFrame(this._tick);
        } else {
          // Fully idle — stop the RAF loop entirely
          this.isRunning = false;
          if (this.app) this.app.updateFps(this.fps, 0); // forces "(settled)" label
        }
      }
    }

    _fitToBounds() {
      // Frame all nodes in the viewport regardless of where forces left them.
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (const n of this.nodes) {
        if (n.x < minX) minX = n.x;
        if (n.y < minY) minY = n.y;
        if (n.x > maxX) maxX = n.x;
        if (n.y > maxY) maxY = n.y;
      }
      if (!isFinite(minX)) return;
      const w = this._W(), h = this._H();
      const bw = Math.max(maxX - minX, 1), bh = Math.max(maxY - minY, 1);
      const pad = 60;
      const zx = w / (bw + pad * 2), zy = h / (bh + pad * 2);
      let z = Math.min(zx, zy, 4);
      if (!isFinite(z) || z <= 0) z = 1;
      const bcx = (minX + maxX) / 2, bcy = (minY + maxY) / 2;
      this.targetZoom = z;
      this.targetPanX = w / 2 - bcx * z;
      this.targetPanY = h / 2 - bcy * z;
      this._needRedraw = true;
    }

    _applyForces(a) {
      const cx = this._W() / 2;
      const cy = this._H() / 2;

      // 1. Center gravity for memories (strong enough to counter repulsion)
      const centerForce = 0.008 * a;
      for (const n of this.memoryNodes) {
        n.vx += (cx - n.x) * centerForce;
        n.vy += (cy - n.y) * centerForce;
      }

      // 1b. Hard bounding constraint — clamp every node inside a sane world
      // radius so a force imbalance can NEVER strand them off-screen.
      const maxR = Math.max(this._W(), this._H()) * 0.85;
      const all = this.nodes;
      for (const n of all) {
        const dx = n.x - cx, dy = n.y - cy;
        const r = Math.sqrt(dx * dx + dy * dy);
        if (r > maxR) {
          const k = maxR / r;
          n.x = cx + dx * k;
          n.y = cy + dy * k;
          n.vx *= 0.5;
          n.vy *= 0.5;
        }
      }

      // 2. Entity repulsion (pairwise — only ~6 entities, O(n²) is fine)
      for (let i = 0; i < this.entityNodes.length; i++) {
        for (let j = i + 1; j < this.entityNodes.length; j++) {
          const ea = this.entityNodes[i], eb = this.entityNodes[j];
          let dx = eb.x - ea.x, dy = eb.y - ea.y;
          let d2 = dx * dx + dy * dy;
          if (d2 < 1) d2 = 1;
          const d = Math.sqrt(d2);
          const f = 2500 * a / d2;
          const fx = (dx / d) * f, fy = (dy / d) * f;
          ea.vx -= fx; ea.vy -= fy;
          eb.vx += fx; eb.vy += fy;
        }
      }

      // 3. Memory → entity repulsion (O(n*m), m ≈ 6, so effectively O(n))
      const memEntRepulsion = 80 * a;
      for (const ni of this.memoryNodes) {
        for (let e = 0; e < this.entityNodes.length; e++) {
          const en = this.entityNodes[e];
          let dx = en.x - ni.x, dy = en.y - ni.y;
          let d2 = dx * dx + dy * dy;
          if (d2 < 1) d2 = 1;
          const d = Math.sqrt(d2);
          const f = memEntRepulsion / d2;
          ni.vx -= (dx / d) * f;
          ni.vy -= (dy / d) * f;
        }
      }

      // 4. Memory → memory repulsion (Barnes-Hut O(n log n))
      const memRepulsion = 120 * a;
      if (this.memoryNodes.length > 1) {
        const qt = this._bhBuild(this.memoryNodes);
        for (const ni of this.memoryNodes) {
          const force = { vx: 0, vy: 0 };
          this._bhForce(qt, ni, memRepulsion, force);
          ni.vx += force.vx;
          ni.vy += force.vy;
        }
      }

      // 5. Link spring force
      const linkDist = 55;
      const linkForce = 0.04 * a;
      for (const e of this.edges) {
        const sn = e.s, tn = e.t;
        if (!sn || !tn) continue;
        let dx = tn.x - sn.x, dy = tn.y - sn.y;
        let d = Math.sqrt(dx * dx + dy * dy);
        if (d < 0.01) d = 0.01;
        const f = (d - linkDist) * linkForce;
        const fx = (dx / d) * f, fy = (dy / d) * f;
        sn.vx += fx; sn.vy += fy;
        tn.vx -= fx; tn.vy -= fy;
      }

      // 6. Apply velocity with damping
      const damping = 0.82;
      for (const node of this.nodes) {
        if (node === this.dragNode) continue;
        node.vx *= damping;
        node.vy *= damping;
        node.x += node.vx;
        node.y += node.vy;
      }
    }

    // ── Rendering (canvas only, no shadowBlur) ────────────────────────

    _draw(time) {
      const ctx = this.ctx;
      const w = this._W();
      const h = this._H();

      // Clear with void background
      ctx.fillStyle = '#06070b';
      ctx.fillRect(0, 0, w, h);

      // Starfield (screen space)
      this.starfield.draw(ctx, w, h, time, this.panX, this.panY);

      // World transform
      ctx.save();
      ctx.translate(this.panX, this.panY);
      ctx.scale(this.zoom, this.zoom);

      // ── Viewport bounds for culling (in world space) ──
      const vx0 = -this.panX / this.zoom;
      const vy0 = -this.panY / this.zoom;
      const vx1 = (w - this.panX) / this.zoom;
      const vy1 = (h - this.panY) / this.zoom;
      const margin = 50; // generous culling margin

      // ── Draw edges (culled) ──────────────────────────────────
      // PERF: batch edges into one path + single stroke. Per-edge stroke()
      // at ~1384 edges was the dominant draw cost (each stroke flushes the
      // rasterizer). During settle we drop to a single uniform-alpha batched
      // path; when settled we batch by coarse alpha-bucket (a few strokes).
      ctx.lineCap = 'round';
      const settling = this.alpha > this.alphaMin;
      if (settling) {
        // Cheapest path: one batched path, uniform alpha, no per-edge work
        ctx.strokeStyle = 'rgba(80,85,100,0.22)';
        ctx.lineWidth = 0.8;
        ctx.beginPath();
        for (const e of this.edges) {
          const sn = e.s, tn = e.t;
          if (!sn || !tn) continue;
          if ((sn.x < vx0 - margin || sn.x > vx1 + margin) &&
              (tn.x < vx0 - margin || tn.x > vx1 + margin)) continue;
          if ((sn.y < vy0 - margin || sn.y > vy1 + margin) &&
              (tn.y < vy0 - margin || tn.y > vy1 + margin)) continue;
          ctx.moveTo(sn.x, sn.y);
          ctx.lineTo(tn.x, tn.y);
        }
        ctx.stroke();
      } else {
        // Settled: batch by coarse alpha-bucket so highlight/search states
        // still show, but only ~4 stroke() calls instead of 1384.
        const buckets = { 0.1: [], 0.3: [], 0.8: [] };
        for (const e of this.edges) {
          const sn = e.s, tn = e.t;
          if (!sn || !tn) continue;
          if ((sn.x < vx0 - margin || sn.x > vx1 + margin) &&
              (tn.x < vx0 - margin || tn.x > vx1 + margin)) continue;
          if ((sn.y < vy0 - margin || sn.y > vy1 + margin) &&
              (tn.y < vy0 - margin || tn.y > vy1 + margin)) continue;
          let a = 0.3;
          if (this.activatedSet) {
            if (this.activatedSet.has(sn.id) && this.activatedSet.has(tn.id)) a = 0.8;
            else if (!this.activatedSet.has(sn.id) && !this.activatedSet.has(tn.id)) a = 0.1;
          }
          if (this.searchDims.size > 0) {
            if (this.searchDims.has(sn.id) || this.searchDims.has(tn.id)) a = Math.max(a * 0.2, 0.06);
          }
          const key = a >= 0.8 ? '0.8' : (a <= 0.1 ? '0.1' : '0.3');
          buckets[key].push(sn.x, sn.y, tn.x, tn.y);
        }
        ctx.lineWidth = 0.8;
        for (const k in buckets) {
          const seg = buckets[k];
          if (!seg.length) continue;
          ctx.strokeStyle = `rgba(80,85,100,${k})`;
          ctx.beginPath();
          for (let i = 0; i < seg.length; i += 4) {
            ctx.moveTo(seg[i], seg[i + 1]);
            ctx.lineTo(seg[i + 2], seg[i + 3]);
          }
          ctx.stroke();
        }
      }

      // ── Draw memory nodes ────────────────────────────────────
      // PERF: use precomputed n.style (brightness+network baked once).
      // During settle we skip per-node alpha math (activation/search/hover
      // are interaction-time only) and skip the contested flicker — pure
      // flat fills, one fillStyle set per node, no per-frame rgba allocs.
      const hasSearch = this.searchDims.size > 0;
      const hasAct = !!this.activatedSet;
      for (const n of this.memoryNodes) {
        // Cull off-screen
        if (n.x < vx0 - margin || n.x > vx1 + margin) continue;
        if (n.y < vy0 - margin || n.y > vy1 + margin) continue;

        if (settling) {
          // Cheapest: flat fill with precomputed style
          ctx.fillStyle = n.style;
          ctx.beginPath();
          ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
          ctx.fill();
          continue;
        }

        // Settled — full fidelity (search/activation/hover/contested)
        let alpha = n.fillAlpha;
        if (hasSearch && this.searchDims.has(n.id)) alpha *= 0.15;
        if (hasAct) {
          if (this.activatedSet.has(n.id)) alpha = Math.max(alpha, 0.9);
          else alpha *= 0.2;
        }
        if (n === this.hoverNode) alpha = Math.max(alpha, 0.9);

        const rgb = NETWORK_RGB[n.network] || [130, 130, 140];
        if (n.status === 'contested') {
          const flicker = Math.sin(time * 0.003 + n.flickerPhase) * 0.3 + 0.7;
          alpha *= flicker;
          ctx.fillStyle = rgba(STATUS_RGB.contested, alpha * 0.8);
          ctx.beginPath();
          ctx.arc(n.x, n.y, n.radius, -Math.PI / 2, Math.PI / 2);
          ctx.fill();
          ctx.fillStyle = rgba(rgb, alpha);
          ctx.beginPath();
          ctx.arc(n.x, n.y, n.radius, Math.PI / 2, -Math.PI / 2);
          ctx.fill();
        } else {
          ctx.fillStyle = rgba(rgb, alpha);
          ctx.beginPath();
          ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
          ctx.fill();
        }

        // Selected node: ring
        if (n === this.selectedNode) {
          ctx.strokeStyle = rgba(rgb, 0.9);
          ctx.lineWidth = 2 / this.zoom;
          ctx.beginPath();
          ctx.arc(n.x, n.y, n.radius + 4, 0, Math.PI * 2);
          ctx.stroke();
        }
      }

      // ── Draw entity nodes ────────────────────────────────────
      for (const n of this.entityNodes) {
        // Cull
        if (n.x < vx0 - margin || n.x > vx1 + margin) continue;
        if (n.y < vy0 - margin || n.y > vy1 + margin) continue;

        const rgb = ENTITY_RGB[n.entity_type] || [130, 130, 140];
        const pulse = Math.sin(time * 0.0008 + n.pulsePhase) * 0.08 + 0.92;
        let alpha = pulse;
        if (n === this.hoverNode) alpha = 1.0;

        if (this.activatedSet) {
          alpha = this.activatedSet.has(n.id) ? 1.0 : alpha * 0.3;
        }

        // Soft radial glow — gradient cached by (r,g,b,alphaBucket) key
        const alphaBucket = Math.round(alpha * 10);
        const gradKey = `${rgb[0]},${rgb[1]},${rgb[2]},${alphaBucket}`;
        let grad = this._gradCache.get(gradKey);
        if (!grad) {
          const glowR = n.radius * 6;
          grad = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, glowR);
          grad.addColorStop(0, rgba(rgb, alpha * 0.4));
          grad.addColorStop(0.4, rgba(rgb, alpha * 0.12));
          grad.addColorStop(1, rgba(rgb, 0));
          this._gradCache.set(gradKey, grad);
        }
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.radius * 6, 0, Math.PI * 2);
        ctx.fill();

        // Outline style: person = dashed, project = solid
        ctx.strokeStyle = rgba(rgb, alpha);
        ctx.lineWidth = 2 / this.zoom;
        if (n.entity_type === 'person' || n.entity_type === 'self') {
          ctx.setLineDash([4 / this.zoom, 3 / this.zoom]);
        } else {
          ctx.setLineDash([]);
        }

        // Filled circle with outline
        ctx.fillStyle = rgba(rgb, alpha * 0.25);
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
        ctx.fill();

        ctx.beginPath();
        ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]); // reset

        // Label (subtle, small, monospace)
        const labelAlpha = (n === this.hoverNode || n === this.selectedNode) ? 0.9 : 0.45;
        ctx.fillStyle = `rgba(190,200,235,${labelAlpha})`;
        ctx.font = `400 ${10 / this.zoom}px "JetBrains Mono", monospace`;
        ctx.textAlign = 'center';
        ctx.fillText(n.label.substring(0, 18), n.x, n.y - n.radius - 8 / this.zoom);

        // Selected entity ring
        if (n === this.selectedNode) {
          ctx.strokeStyle = rgba(rgb, 0.8);
          ctx.lineWidth = 2 / this.zoom;
          ctx.beginPath();
          ctx.arc(n.x, n.y, n.radius + 5, 0, Math.PI * 2);
          ctx.stroke();
        }
      }

      ctx.restore();
    }

    // ── Interaction helpers ──────────────────────────────────────────

    _screenToWorld(sx, sy) {
      return {
        x: (sx - this.panX) / this.zoom,
        y: (sy - this.panY) / this.zoom,
      };
    }

    _findNode(sx, sy) {
      const { x, y } = this._screenToWorld(sx, sy);
      for (let i = this.entityNodes.length - 1; i >= 0; i--) {
        const n = this.entityNodes[i];
        const dx = x - n.x, dy = y - n.y;
        if (dx * dx + dy * dy < (n.radius + 8) * (n.radius + 8)) return n;
      }
      for (let i = this.memoryNodes.length - 1; i >= 0; i--) {
        const n = this.memoryNodes[i];
        const dx = x - n.x, dy = y - n.y;
        if (dx * dx + dy * dy < (n.radius + 5) * (n.radius + 5)) return n;
      }
      return null;
    }

    _onMouseDown(e) {
      const rect = this.canvas.getBoundingClientRect();
      const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
      const node = this._findNode(sx, sy);
      this.dragStartX = sx;
      this.dragStartY = sy;
      if (node) {
        this.dragNode = node;
        this.isDragging = false;
      } else {
        this.isPanning = true;
      }
      this.lastX = sx;
      this.lastY = sy;
    }

    _onMouseMove(e) {
      const rect = this.canvas.getBoundingClientRect();
      const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
      const dragDist = Math.abs(sx - this.dragStartX) + Math.abs(sy - this.dragStartY);

      if (this.dragNode && dragDist > 3) this.isDragging = true;

      if (this.dragNode) {
        const { x, y } = this._screenToWorld(sx, sy);
        this.dragNode.x = x;
        this.dragNode.y = y;
        this.dragNode.vx = 0;
        this.dragNode.vy = 0;
        this._restartAlpha(0.2);
      } else if (this.isPanning) {
        this.targetPanX += sx - this.lastX;
        this.targetPanY += sy - this.lastY;
        this.panX = this.targetPanX;
        this.panY = this.targetPanY;
        this.lastX = sx;
        this.lastY = sy;
        this._needRedraw = true;
        if (!this.isRunning) this._start();
      } else {
        const node = this._findNode(sx, sy);
        if (node !== this.hoverNode) {
          this.hoverNode = node;
          this.canvas.style.cursor = node ? 'pointer' : 'grab';
          this._needRedraw = true;
          if (!this.isRunning) this._start();
        }

        // Tooltip
        if (this.app) {
          if (node) {
            this.app.showTooltip(sx, sy, node);
          } else {
            this.app.hideTooltip();
          }
        }
      }
    }

    _onMouseUp(e) {
      if (this.dragNode && !this.isDragging) {
        this.selectNode(this.dragNode);
      }
      this.dragNode = null;
      this.isDragging = false;
      this.isPanning = false;
    }

    _onMouseLeave() {
      this.dragNode = null;
      this.isDragging = false;
      this.isPanning = false;
      this.hoverNode = null;
      if (this.app) this.app.hideTooltip();
      this._needRedraw = true;
    }

    _onWheel(e) {
      e.preventDefault();
      const rect = this.canvas.getBoundingClientRect();
      const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
      const delta = e.deltaY > 0 ? 0.88 : 1.12;
      const newZoom = Math.max(0.15, Math.min(8, this.targetZoom * delta));
      this.targetPanX = sx - (sx - this.targetPanX) * (newZoom / this.targetZoom);
      this.targetPanY = sy - (sy - this.targetPanY) * (newZoom / this.targetZoom);
      this.targetZoom = newZoom;
      this._needRedraw = true;
      if (!this.isRunning) this._start();
    }

    // Touch
    _onTouchStart(e) {
      e.preventDefault();
      if (e.touches.length === 1) {
        const rect = this.canvas.getBoundingClientRect();
        const sx = e.touches[0].clientX - rect.left;
        const sy = e.touches[0].clientY - rect.top;
        const node = this._findNode(sx, sy);
        if (node) {
          this.dragNode = node;
          this.dragStartX = sx;
          this.dragStartY = sy;
        } else {
          this.isPanning = true;
        }
        this.lastX = sx;
        this.lastY = sy;
      } else if (e.touches.length === 2) {
        this._touchPinchDist = this._pinchDist(e);
      }
    }

    _pinchDist(e) {
      const dx = e.touches[0].clientX - e.touches[1].clientX;
      const dy = e.touches[0].clientY - e.touches[1].clientY;
      return Math.sqrt(dx * dx + dy * dy);
    }

    _onTouchMove(e) {
      e.preventDefault();
      if (e.touches.length === 1 && this.dragNode) {
        const rect = this.canvas.getBoundingClientRect();
        const sx = e.touches[0].clientX - rect.left;
        const sy = e.touches[0].clientY - rect.top;
        const { x, y } = this._screenToWorld(sx, sy);
        this.dragNode.x = x;
        this.dragNode.y = y;
        this.dragNode.vx = 0;
        this.dragNode.vy = 0;
        this._restartAlpha(0.2);
      } else if (e.touches.length === 1 && this.isPanning) {
        const rect = this.canvas.getBoundingClientRect();
        const sx = e.touches[0].clientX - rect.left;
        const sy = e.touches[0].clientY - rect.top;
        this.targetPanX += sx - this.lastX;
        this.targetPanY += sy - this.lastY;
        this.panX = this.targetPanX;
        this.panY = this.targetPanY;
        this.lastX = sx;
        this.lastY = sy;
        this._needRedraw = true;
        if (!this.isRunning) this._start();
      } else if (e.touches.length === 2 && this._touchPinchDist) {
        const nd = this._pinchDist(e);
        const scale = nd / this._touchPinchDist;
        this.targetZoom = Math.max(0.15, Math.min(8, this.targetZoom * scale));
        this._touchPinchDist = nd;
        this._needRedraw = true;
        if (!this.isRunning) this._start();
      }
    }

    _onTouchEnd(e) {
      if (this.dragNode && e.touches.length === 0) {
        const rect = this.canvas.getBoundingClientRect();
        const dx = Math.abs(this.dragStartX - (this.dragNode.x * this.zoom + this.panX));
        const dy = Math.abs(this.dragStartY - (this.dragNode.y * this.zoom + this.panY));
        if (dx < 10 && dy < 10) {
          this.selectNode(this.dragNode);
        }
      }
      this.dragNode = null;
      this.isPanning = false;
      this._touchPinchDist = 0;
    }

    // ── Node selection + spreading activation ────────────────────────

    selectNode(node) {
      this.selectedNode = node;

      // Compute spreading activation neighborhood
      if (node) {
        this.activatedSet = new Set([node.id]);
        // 1-hop neighbors via edges
        for (const e of this.edges) {
          if (e.s === node) this.activatedSet.add(e.t.id);
          if (e.t === node) this.activatedSet.add(e.s.id);
        }
        // If entity: all its memories are already connected via shared_entity edges
      } else {
        this.activatedSet = null;
      }

      this._needRedraw = true;
      if (!this.isRunning) this._start();

      // Fire detail callback
      if (this.app) {
        this.app.onSelectNode(node);
      }
    }

    // ── Search dim ───────────────────────────────────────────────────

    applySearch(query) {
      this.searchQuery = query;
      if (!query || query.trim().length === 0) {
        this.searchDims.clear();
        this._needRedraw = true;
        if (!this.isRunning) this._start();
        return;
      }

      const q = query.toLowerCase();
      this.searchDims.clear();
      for (const n of this.nodes) {
        const label = (n.label || '').toLowerCase();
        const text = (n.text || '').toLowerCase();
        if (!label.includes(q) && !text.includes(q)) {
          this.searchDims.add(n.id);
        }
      }
      this._needRedraw = true;
      if (!this.isRunning) this._start();
    }

    // ── Zoom controls ────────────────────────────────────────────────

    zoomBy(factor) {
      const cx = this._W() / 2;
      const cy = this._H() / 2;
      const newZoom = Math.max(0.15, Math.min(8, this.targetZoom * factor));
      this.targetPanX = cx - (cx - this.targetPanX) * (newZoom / this.targetZoom);
      this.targetPanY = cy - (cy - this.targetPanY) * (newZoom / this.targetZoom);
      this.targetZoom = newZoom;
      this._needRedraw = true;
      if (!this.isRunning) this._start();
    }
  }

  // Export
  window.GalaxyView = GalaxyView;
})();
