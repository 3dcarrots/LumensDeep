/* ==========================================================================
   Lumens Deep — Official Website Interactive Engine
   Domain: https://lumensdeep.com/
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
  initMobileMenu();
  initSplitViewer();
  initMapShowcase();
  initBokehPlayground();
  initToneSplineDemo();
  initColorEqualizerDemo();
  initCopyButtons();
  initDocsSearch();
  initShortcutsFilter();
});

function initMobileMenu() {
  const mobileBtn = document.getElementById('mobile-menu-btn');
  const navLinks = document.querySelector('.nav-links');
  if (mobileBtn && navLinks) {
    mobileBtn.addEventListener('click', () => {
      navLinks.classList.toggle('open');
    });
    navLinks.querySelectorAll('a').forEach(link => {
      link.addEventListener('click', () => {
        navLinks.classList.remove('open');
      });
    });
  }
}

/* --------------------------------------------------------------------------
   1. Interactive Before/After Split Viewer
   -------------------------------------------------------------------------- */
function initSplitViewer() {
  const containers = document.querySelectorAll('.split-viewer-container');
  if (!containers || containers.length === 0) return;

  containers.forEach(container => {
    const afterWrapper = container.querySelector('.split-image-after-wrapper');
    const dividerLine = container.querySelector('.split-divider-line');
    const handleKnob = container.querySelector('.split-handle-knob') || container.querySelector('.split-handle');

    if (!afterWrapper || !dividerLine || !handleKnob) return;

    let isDragging = false;

    function updateSplitPosition(clientX) {
      const rect = container.getBoundingClientRect();
      let posX = clientX - rect.left;
      let percentage = (posX / rect.width) * 100;
      percentage = Math.max(1, Math.min(99, percentage));

      afterWrapper.style.clipPath = `polygon(${percentage}% 0, 100% 0, 100% 100%, ${percentage}% 100%)`;
      dividerLine.style.left = `${percentage}%`;
      handleKnob.style.left = `${percentage}%`;
    }

    container.addEventListener('mousedown', (e) => {
      isDragging = true;
      updateSplitPosition(e.clientX);
    });

    window.addEventListener('mousemove', (e) => {
      if (!isDragging) return;
      updateSplitPosition(e.clientX);
    });

    window.addEventListener('mouseup', () => {
      isDragging = false;
    });

    // Touch Support
    container.addEventListener('touchstart', (e) => {
      isDragging = true;
      if (e.touches.length > 0) {
        updateSplitPosition(e.touches[0].clientX);
      }
    }, { passive: true });

    window.addEventListener('touchmove', (e) => {
      if (!isDragging || e.touches.length === 0) return;
      updateSplitPosition(e.touches[0].clientX);
    }, { passive: true });

    window.addEventListener('touchend', () => {
      isDragging = false;
    });
  });
}

/* --------------------------------------------------------------------------
   2. Interactive 3D Neural Maps Showcase
   -------------------------------------------------------------------------- */
function initMapShowcase() {
  const tabBtns = document.querySelectorAll('.map-tab-btn');
  const displayImg = document.getElementById('map-display-img');
  const infoTitle = document.getElementById('map-info-title');
  const infoDesc = document.getElementById('map-info-desc');
  const keyShortcut = document.getElementById('map-key-shortcut');

  if (!tabBtns.length || !displayImg) return;

  const mapData = {
    original: {
      img: 'assets/images/hero_after.jpg',
      title: 'Full 32-Bit Linear RGB Render',
      desc: 'The complete non-destructive render with custom OKLCH tone curves, color balance, and hardware-accelerated ModernGL layer accumulation.',
      shortcut: 'Original Output'
    },
    depth: {
      img: 'assets/images/sample_depth.jpg',
      title: 'High-Resolution 3D Depth Map',
      desc: 'Estimated offline using neural metric depth transformers. Enables realistic optical depth-of-field raymarching and spatial foreground/background masking.',
      shortcut: 'Keyboard: D'
    },
    normals: {
      img: 'assets/images/sample_normals.jpg',
      title: 'Surface Normal Orientation Map',
      desc: 'High-frequency surface vectors (XYZ encoded in RGB). Allows directional 3D scene relighting, angle-based masking, and specular highlight shaping.',
      shortcut: 'Keyboard: N'
    },
    subject: {
      img: 'assets/images/sample_subject.jpg',
      title: 'AI Alpha Matting & Subject Mask',
      desc: 'Sub-pixel hairline segmentation and alpha matting for instant subject separation, background relighting, and isolated color grading.',
      shortcut: 'Keyboard: S'
    }
  };

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const type = btn.getAttribute('data-map-type');
      const data = mapData[type];
      if (data) {
        displayImg.style.opacity = '0';
        setTimeout(() => {
          displayImg.src = data.img;
          if (infoTitle) infoTitle.textContent = data.title;
          if (infoDesc) infoDesc.textContent = data.desc;
          if (keyShortcut) keyShortcut.innerHTML = `<span class="kbd-badge">${data.shortcut}</span>`;
          displayImg.style.opacity = '1';
        }, 150);
      }
    });
  });
}

/* --------------------------------------------------------------------------
   3. Interactive Bokeh Aperture Blades Geometry Visualizer
   -------------------------------------------------------------------------- */
function initBokehPlayground() {
  const canvas = document.getElementById('aperture-canvas');
  const bladeButtons = document.querySelectorAll('.blade-btn');
  const bladeLabel = document.getElementById('blade-shape-name');

  if (!canvas || !bladeButtons.length) return;

  const ctx = canvas.getContext('2d');
  let currentBlades = 5;

  function drawAperture(numBlades) {
    const w = canvas.width = 300;
    const h = canvas.height = 300;
    ctx.clearRect(0, 0, w, h);

    const cx = w / 2;
    const cy = h / 2;
    const radius = 95;

    // Background lens circle
    ctx.beginPath();
    ctx.arc(cx, cy, 120, 0, Math.PI * 2);
    ctx.fillStyle = '#0e1420';
    ctx.fill();
    ctx.lineWidth = 4;
    ctx.strokeStyle = 'rgba(0, 210, 255, 0.2)';
    ctx.stroke();

    // Draw Aperture Pupil Disc
    ctx.beginPath();
    if (numBlades >= 10) {
      // Circular smooth iris
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    } else {
      // Regular Polygon
      for (let i = 0; i < numBlades; i++) {
        const angle = (i * 2 * Math.PI) / numBlades - Math.PI / 2;
        const x = cx + radius * Math.cos(angle);
        const y = cy + radius * Math.sin(angle);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
    }

    // Glow Gradient Fill
    const grad = ctx.createRadialGradient(cx, cy, 10, cx, cy, radius);
    grad.addColorStop(0, 'rgba(0, 242, 254, 0.9)');
    grad.addColorStop(0.7, 'rgba(0, 210, 255, 0.6)');
    grad.addColorStop(1, 'rgba(138, 43, 226, 0.3)');

    ctx.fillStyle = grad;
    ctx.fill();

    ctx.lineWidth = 3;
    ctx.strokeStyle = '#00d2ff';
    ctx.shadowColor = '#00d2ff';
    ctx.shadowBlur = 15;
    ctx.stroke();
    ctx.shadowBlur = 0; // reset
  }

  const shapeNames = {
    3: '3 Blades — Triangle (Sharp Geometric Bokeh)',
    4: '4 Blades — Square (Vintage Anamorphic Look)',
    5: '5 Blades — Pentagon (Classic Hollywood Bokeh)',
    6: '6 Blades — Hexagon (Standard Prime Lens)',
    8: '8 Blades — Octagon (Premium Portrait Iris)',
    10: 'Smooth Circular Iris (Creamy Modern Bokeh)'
  };

  bladeButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      bladeButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentBlades = parseInt(btn.getAttribute('data-blades'), 10);
      drawAperture(currentBlades);
      if (bladeLabel) {
        bladeLabel.textContent = shapeNames[currentBlades] || `${currentBlades} Blades`;
      }
    });
  });

  drawAperture(currentBlades);
}

/* --------------------------------------------------------------------------
   4. Interactive 8-Band OKLCH Tone Equalizer Curve Sandbox & Live WebGL Photo
   -------------------------------------------------------------------------- */
function initToneSplineDemo() {
  const canvas = document.getElementById('tone-spline-canvas');
  const glCanvas = document.getElementById('tone-webgl-canvas');
  if (!canvas) return;

  // --- 1. Spline Curve Canvas Setup ---
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const logicalWidth = 600;
  const logicalHeight = 240;

  canvas.width = logicalWidth * dpr;
  canvas.height = logicalHeight * dpr;
  ctx.scale(dpr, dpr);

  const w = logicalWidth;
  const h = logicalHeight;

  // 8 Luminance Zones from deep shadow to specular highlight
  const zoneLabels = ["-8 EV", "-6 EV", "-4 EV", "-2 EV", "-1 EV", "0 EV", "+1 EV", "+2 EV"];
  let points = [0, 0, 0, 0.25, 0.45, 0.2, 0, 0]; // default subtle curve

  const padX = 42;
  const padY = 38;
  const midY = (h - padY) / 2 + 10;
  const scaleEV = (midY - padY) / 1.5; // +-1.5 EV max
  const step = (w - 2 * padX) / (points.length - 1);

  let activeIdx = null;
  let hoveredIdx = null;
  let isDragging = false;
  let isShowingOriginal = false;

  // --- 2. WebGL Photo Renderer Setup ---
  let gl = null;
  let glProgram = null;
  let glTexture = null;
  let uPointsLoc = null;
  let uIsOrigLoc = null;
  let photoImg = new Image();

  if (glCanvas) {
    gl = glCanvas.getContext('webgl') || glCanvas.getContext('experimental-webgl');
  }

  if (gl) {
    const vsSource = `
      attribute vec2 a_pos;
      attribute vec2 a_uv;
      varying vec2 v_uv;
      void main() {
        v_uv = a_uv;
        gl_Position = vec4(a_pos, 0.0, 1.0);
      }
    `;

    const fsSource = `
      precision mediump float;
      varying vec2 v_uv;
      uniform sampler2D u_image;
      uniform float u_points[8];
      uniform float u_is_original;

      vec3 srgb_to_linear(vec3 c) {
        return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
      }

      vec3 linear_to_srgb(vec3 c) {
        c = max(c, vec3(0.0));
        return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
      }

      float eval_spline(float ev) {
        float zone_ev[8];
        zone_ev[0] = -8.0;
        zone_ev[1] = -6.0;
        zone_ev[2] = -4.0;
        zone_ev[3] = -2.0;
        zone_ev[4] = -1.0;
        zone_ev[5] =  0.0;
        zone_ev[6] =  1.0;
        zone_ev[7] =  2.0;

        if (ev <= zone_ev[0]) return u_points[0];
        if (ev >= zone_ev[7]) return u_points[7];

        for (int i = 0; i < 7; i++) {
          if (ev >= zone_ev[i] && ev <= zone_ev[i+1]) {
            float t = (ev - zone_ev[i]) / (zone_ev[i+1] - zone_ev[i]);
            float p0 = (i > 0) ? u_points[i-1] : (2.0 * u_points[0] - u_points[1]);
            float p1 = u_points[i];
            float p2 = u_points[i+1];
            float p3 = (i < 6) ? u_points[i+2] : (2.0 * u_points[7] - u_points[6]);

            float t2 = t * t;
            float t3 = t2 * t;
            return 0.5 * (
              (2.0 * p1) +
              (-p0 + p2) * t +
              (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
              (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            );
          }
        }
        return 0.0;
      }

      void main() {
        vec4 src = texture2D(u_image, v_uv);
        if (u_is_original > 0.5) {
          gl_FragColor = src;
          return;
        }
        vec3 linear_rgb = srgb_to_linear(src.rgb);
        float luma = dot(linear_rgb, vec3(0.2126, 0.7152, 0.0722));
        float ev = log2(max(luma, 0.00001));

        float ev_shift = eval_spline(ev);
        vec3 corrected = linear_rgb * exp2(ev_shift);
        vec3 result = linear_to_srgb(corrected);

        gl_FragColor = vec4(result, src.a);
      }
    `;

    function createShader(type, src) {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      return s;
    }

    const vs = createShader(gl.VERTEX_SHADER, vsSource);
    const fs = createShader(gl.FRAGMENT_SHADER, fsSource);
    glProgram = gl.createProgram();
    gl.attachShader(glProgram, vs);
    gl.attachShader(glProgram, fs);
    gl.linkProgram(glProgram);
    gl.useProgram(glProgram);

    // Quad geometry (inverted UV Y for WebGL texture orientation)
    const quad = new Float32Array([
      -1, -1,  0, 1,
       1, -1,  1, 1,
      -1,  1,  0, 0,
      -1,  1,  0, 0,
       1, -1,  1, 1,
       1,  1,  1, 0,
    ]);

    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);

    const aPos = gl.getAttribLocation(glProgram, 'a_pos');
    const aUv = gl.getAttribLocation(glProgram, 'a_uv');
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 16, 0);
    gl.enableVertexAttribArray(aUv);
    gl.vertexAttribPointer(aUv, 2, gl.FLOAT, false, 16, 8);

    uPointsLoc = gl.getUniformLocation(glProgram, 'u_points');
    uIsOrigLoc = gl.getUniformLocation(glProgram, 'u_is_original');

    glTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, glTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    function loadSamplePhoto(srcUrl) {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        try {
          gl.bindTexture(gl.TEXTURE_2D, glTexture);
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
          renderWebGL();
        } catch (e) {
          // Fallback if browser enforces strict local file:// security
          const imgFallback = new Image();
          imgFallback.onload = () => {
            try {
              gl.bindTexture(gl.TEXTURE_2D, glTexture);
              gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, imgFallback);
              renderWebGL();
            } catch (err) {}
          };
          imgFallback.src = srcUrl;
        }
      };
      img.onerror = () => {
        const imgFallback = new Image();
        imgFallback.onload = () => {
          try {
            gl.bindTexture(gl.TEXTURE_2D, glTexture);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, imgFallback);
            renderWebGL();
          } catch (err) {}
        };
        imgFallback.src = srcUrl;
      };
      img.src = srcUrl;
    }

    loadSamplePhoto(window.SAMPLE_LANDSCAPE_B64 || 'assets/images/sample_landscape.jpg');

    // Photo Sample Switches
    const btnSample1 = document.getElementById('btn-photo-sample-1');
    const btnSample2 = document.getElementById('btn-photo-sample-2');
    if (btnSample1) {
      btnSample1.addEventListener('click', () => {
        btnSample1.classList.add('btn-primary');
        btnSample1.classList.remove('btn-secondary');
        if (btnSample2) {
          btnSample2.classList.add('btn-secondary');
          btnSample2.classList.remove('btn-primary');
        }
        loadSamplePhoto(window.SAMPLE_LANDSCAPE_B64 || 'assets/images/sample_landscape.jpg');
      });
    }
    if (btnSample2) {
      btnSample2.addEventListener('click', () => {
        btnSample2.classList.add('btn-primary');
        btnSample2.classList.remove('btn-secondary');
        if (btnSample1) {
          btnSample1.classList.add('btn-secondary');
          btnSample1.classList.remove('btn-primary');
        }
        loadSamplePhoto(window.SAMPLE_PORTRAIT_B64 || 'assets/images/sample_portrait.jpg');
      });
    }

    // Compare button (Hold to view original)
    const btnCompare = document.getElementById('btn-tone-compare');
    if (btnCompare) {
      const showOrig = (e) => { e.preventDefault(); isShowingOriginal = true; renderWebGL(); };
      const hideOrig = (e) => { e.preventDefault(); isShowingOriginal = false; renderWebGL(); };
      btnCompare.addEventListener('mousedown', showOrig);
      btnCompare.addEventListener('touchstart', showOrig, { passive: false });
      btnCompare.addEventListener('mouseup', hideOrig);
      btnCompare.addEventListener('mouseleave', hideOrig);
      btnCompare.addEventListener('touchend', hideOrig);
    }
  }

  function renderWebGL() {
    if (!gl || !glProgram) return;
    gl.viewport(0, 0, glCanvas.width, glCanvas.height);
    gl.useProgram(glProgram);
    gl.uniform1fv(uPointsLoc, new Float32Array(points));
    gl.uniform1f(uIsOrigLoc, isShowingOriginal ? 1.0 : 0.0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  function getCanvasCoords(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = w / rect.width;
    const scaleY = h / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    };
  }

  // Smooth Catmull-Rom Spline Interpolation
  function catmullRom(p0, p1, p2, p3, t) {
    const t2 = t * t;
    const t3 = t2 * t;
    return 0.5 * (
      (2 * p1) +
      (-p0 + p2) * t +
      (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
      (-p0 + 3 * p1 - 3 * p2 + p3) * t3
    );
  }

  function getCurvePoints() {
    const pts = [];
    const n = points.length;
    const coords = [];

    for (let i = 0; i < n; i++) {
      coords.push({
        x: padX + i * step,
        y: midY - points[i] * scaleEV
      });
    }

    const pVirtualBefore = {
      x: coords[0].x - step,
      y: coords[0].y - (coords[1].y - coords[0].y)
    };
    const pVirtualAfter = {
      x: coords[n - 1].x + step,
      y: coords[n - 1].y + (coords[n - 1].y - coords[n - 2].y)
    };

    const fullCoords = [pVirtualBefore, ...coords, pVirtualAfter];

    const samplesPerSegment = 24;
    for (let i = 0; i < n - 1; i++) {
      const p0 = fullCoords[i];
      const p1 = fullCoords[i + 1];
      const p2 = fullCoords[i + 2];
      const p3 = fullCoords[i + 3];

      for (let s = 0; s < samplesPerSegment; s++) {
        const t = s / samplesPerSegment;
        pts.push({
          x: catmullRom(p0.x, p1.x, p2.x, p3.x, t),
          y: catmullRom(p0.y, p1.y, p2.y, p3.y, t)
        });
      }
    }
    pts.push(coords[n - 1]);
    return pts;
  }

  function drawSpline() {
    ctx.clearRect(0, 0, w, h);

    // 1. Dark Background Fill
    ctx.fillStyle = '#0a0e14';
    ctx.fillRect(0, 0, w, h);

    // 2. Zone Columns & Vertical Grid Lines
    for (let i = 0; i < points.length; i++) {
      const x = padX + i * step;

      // Vertical line
      ctx.strokeStyle = (hoveredIdx === i || activeIdx === i) ? 'rgba(0, 210, 255, 0.25)' : 'rgba(255, 255, 255, 0.05)';
      ctx.lineWidth = (hoveredIdx === i || activeIdx === i) ? 1.5 : 1;
      ctx.beginPath();
      ctx.moveTo(x, padY - 10);
      ctx.lineTo(x, h - padY + 12);
      ctx.stroke();

      // Zone Label at Bottom
      ctx.fillStyle = (hoveredIdx === i || activeIdx === i) ? '#00d2ff' : '#6e7681';
      ctx.font = (hoveredIdx === i || activeIdx === i) ? '600 11px JetBrains Mono, monospace' : '400 11px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(zoneLabels[i], x, h - 10);
    }

    // 3. Horizontal Grid & 0 EV Baseline
    const evLevels = [1.0, 0.5, 0.0, -0.5, -1.0];
    evLevels.forEach(ev => {
      const y = midY - ev * scaleEV;
      ctx.beginPath();
      ctx.moveTo(padX - 10, y);
      ctx.lineTo(w - padX + 10, y);

      if (ev === 0.0) {
        ctx.strokeStyle = 'rgba(0, 210, 255, 0.4)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
      } else {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
        ctx.lineWidth = 1;
        ctx.setLineDash([]);
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // Baseline EV Label
      ctx.fillStyle = ev === 0.0 ? '#00d2ff' : '#484f58';
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.textAlign = 'right';
      ctx.fillText((ev > 0 ? `+${ev.toFixed(1)}` : `${ev.toFixed(1)}`) + ' EV', padX - 14, y + 3);
    });

    // 4. Spline Fill to 0 EV Baseline
    const curvePts = getCurvePoints();
    if (curvePts.length > 0) {
      ctx.beginPath();
      ctx.moveTo(curvePts[0].x, midY);
      curvePts.forEach(pt => ctx.lineTo(pt.x, pt.y));
      ctx.lineTo(curvePts[curvePts.length - 1].x, midY);
      ctx.closePath();

      const grad = ctx.createLinearGradient(0, padY, 0, h - padY);
      grad.addColorStop(0, 'rgba(0, 210, 255, 0.25)');
      grad.addColorStop(0.5, 'rgba(0, 210, 255, 0.05)');
      grad.addColorStop(1, 'rgba(0, 210, 255, 0.2)');
      ctx.fillStyle = grad;
      ctx.fill();

      // 5. Glowing Spline Line
      ctx.beginPath();
      curvePts.forEach((pt, idx) => {
        if (idx === 0) ctx.moveTo(pt.x, pt.y);
        else ctx.lineTo(pt.x, pt.y);
      });
      ctx.lineWidth = 3;
      ctx.strokeStyle = '#00d2ff';
      ctx.shadowColor = 'rgba(0, 210, 255, 0.8)';
      ctx.shadowBlur = 12;
      ctx.stroke();
      ctx.shadowBlur = 0;
    }

    // 6. Control Point Knobs & Tooltip Badges
    for (let i = 0; i < points.length; i++) {
      const px = padX + i * step;
      const py = midY - points[i] * scaleEV;
      const isPointActive = (activeIdx === i);
      const isPointHovered = (hoveredIdx === i);

      // Outer glow circle
      if (isPointActive || isPointHovered) {
        ctx.beginPath();
        ctx.arc(px, py, 12, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0, 210, 255, 0.25)';
        ctx.fill();
      }

      // Point knob
      ctx.beginPath();
      ctx.arc(px, py, isPointActive ? 7 : (isPointHovered ? 6 : 5), 0, Math.PI * 2);
      ctx.fillStyle = isPointActive ? '#00d2ff' : (isPointHovered ? '#38bdf8' : '#f59e0b');
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#ffffff';
      ctx.stroke();

      // EV Value Tooltip Badge over active / hovered point
      if (isPointActive || isPointHovered || Math.abs(points[i]) > 0.05) {
        const val = points[i];
        const text = (val >= 0 ? `+${val.toFixed(2)}` : `${val.toFixed(2)}`) + ' EV';
        ctx.font = '600 10px JetBrains Mono, monospace';
        const tw = ctx.measureText(text).width;
        const badgeW = tw + 10;
        const badgeH = 18;
        const badgeY = py < midY ? py - 22 : py + 14;

        ctx.fillStyle = isPointActive ? 'rgba(0, 210, 255, 0.9)' : 'rgba(21, 27, 35, 0.9)';
        ctx.beginPath();
        ctx.roundRect(px - badgeW / 2, badgeY - badgeH / 2, badgeW, badgeH, 4);
        ctx.fill();
        ctx.strokeStyle = isPointActive ? '#00d2ff' : 'rgba(255, 255, 255, 0.2)';
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.fillStyle = isPointActive ? '#05080e' : '#e6edf3';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, px, badgeY);
        ctx.textBaseline = 'alphabetic';
      }
    }

    // Render WebGL Texture synchronously on GPU!
    renderWebGL();
  }

  drawSpline();

  function findClosestIndex(canvasX) {
    let bestIdx = 0;
    let minDiff = Infinity;
    for (let i = 0; i < points.length; i++) {
      const px = padX + i * step;
      const diff = Math.abs(canvasX - px);
      if (diff < minDiff) {
        minDiff = diff;
        bestIdx = i;
      }
    }
    return bestIdx;
  }

  // Pointer Down (Mouse, Touch, Pen)
  canvas.addEventListener('pointerdown', (e) => {
    const coords = getCanvasCoords(e);
    const idx = findClosestIndex(coords.x);
    activeIdx = idx;
    isDragging = true;
    canvas.setPointerCapture(e.pointerId);

    const newEV = (midY - coords.y) / scaleEV;
    points[activeIdx] = Math.max(-1.5, Math.min(1.5, Math.abs(newEV) < 0.04 ? 0 : newEV));
    drawSpline();
  });

  // Pointer Move
  canvas.addEventListener('pointermove', (e) => {
    const coords = getCanvasCoords(e);
    if (isDragging && activeIdx !== null) {
      const newEV = (midY - coords.y) / scaleEV;
      points[activeIdx] = Math.max(-1.5, Math.min(1.5, Math.abs(newEV) < 0.04 ? 0 : newEV));
      drawSpline();
    } else {
      const prevHover = hoveredIdx;
      const idx = findClosestIndex(coords.x);
      const px = padX + idx * step;
      if (Math.abs(coords.x - px) <= step * 0.55) {
        hoveredIdx = idx;
        canvas.style.cursor = 'ns-resize';
      } else {
        hoveredIdx = null;
        canvas.style.cursor = 'crosshair';
      }
      if (prevHover !== hoveredIdx) {
        drawSpline();
      }
    }
  });

  // Pointer Up / Cancel
  function handlePointerEnd(e) {
    if (isDragging) {
      isDragging = false;
      activeIdx = null;
      try {
        canvas.releasePointerCapture(e.pointerId);
      } catch (err) {}
      drawSpline();
    }
  }

  canvas.addEventListener('pointerup', handlePointerEnd);
  canvas.addEventListener('pointercancel', handlePointerEnd);

  // Double click to reset point / curve
  canvas.addEventListener('dblclick', (e) => {
    const coords = getCanvasCoords(e);
    const idx = findClosestIndex(coords.x);
    const px = padX + idx * step;
    if (Math.abs(coords.x - px) <= step * 0.5) {
      points[idx] = 0;
    } else {
      points = [0, 0, 0, 0, 0, 0, 0, 0];
    }
    drawSpline();
  });

  // Presets
  const presetsContainer = document.getElementById('tone-spline-presets');
  const presetLabel = document.getElementById('tone-active-preset-name');
  if (presetsContainer) {
    presetsContainer.addEventListener('click', (e) => {
      const btn = e.target.closest('button[data-preset]');
      if (!btn) return;
      const preset = btn.getAttribute('data-preset');
      if (preset === 'flat') {
        points = [0, 0, 0, 0, 0, 0, 0, 0];
        if (presetLabel) presetLabel.textContent = 'Flat (0 EV)';
      } else if (preset === 'shadow-lift') {
        points = [0.75, 0.60, 0.40, 0.20, 0.05, 0, 0, 0];
        if (presetLabel) presetLabel.textContent = 'Shadow Lift (+0.75 EV)';
      } else if (preset === 's-curve') {
        points = [-0.45, -0.35, -0.15, 0.0, 0.15, 0.35, 0.50, 0.30];
        if (presetLabel) presetLabel.textContent = 'S-Curve Film Tone';
      } else if (preset === 'highlight-recovery') {
        points = [0, 0, 0, 0, -0.10, -0.30, -0.65, -0.50];
        if (presetLabel) presetLabel.textContent = 'Highlight Recovery (-0.65 EV)';
      }
      drawSpline();
    });
  }
}

/* --------------------------------------------------------------------------
   4b. Interactive OKLCH Color Equalizer (Saturation / Brightness / Hue) WebGL Studio
   -------------------------------------------------------------------------- */
function initColorEqualizerDemo() {
  const canvas = document.getElementById('color-spline-canvas');
  const glCanvas = document.getElementById('color-webgl-canvas');
  if (!canvas) return;

  // --- 1. Spline Curve Canvas Setup ---
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const logicalWidth = 600;
  const logicalHeight = 240;

  canvas.width = logicalWidth * dpr;
  canvas.height = logicalHeight * dpr;
  ctx.scale(dpr, dpr);

  const w = logicalWidth;
  const h = logicalHeight;

  const hueAngles = [0, 60, 120, 180, 240, 300, 360];
  const hueNames = ["Red (0°)", "Yellow (60°)", "Green (120°)", "Cyan (180°)", "Blue (240°)", "Magenta (300°)", "Red (360°)"];

  let currentMode = 'saturation'; // 'saturation' | 'brightness' | 'hue'
  
  const satNodes = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
  const briNodes = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
  const hueNodes = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

  const padX = 42;
  const padY = 38;
  const midY = (h - padY) / 2 + 10;
  const step = (w - 2 * padX) / (hueAngles.length - 1);

  let activeIdx = null;
  let hoveredIdx = null;
  let isDragging = false;
  let isShowingOriginal = false;

  // --- 2. WebGL Photo Renderer Setup ---
  let gl = null;
  let glProgram = null;
  let glTexture = null;
  let uSatLoc = null;
  let uBriLoc = null;
  let uHueLoc = null;
  let uIsOrigLoc = null;
  let photoImg = new Image();

  if (glCanvas) {
    gl = glCanvas.getContext('webgl') || glCanvas.getContext('experimental-webgl');
  }

  if (gl) {
    const vsSource = `
      attribute vec2 a_pos;
      attribute vec2 a_uv;
      varying vec2 v_uv;
      void main() {
        v_uv = a_uv;
        gl_Position = vec4(a_pos, 0.0, 1.0);
      }
    `;

    const fsSource = `
      precision mediump float;
      varying vec2 v_uv;
      uniform sampler2D u_image;
      uniform float u_sat[7];
      uniform float u_bri[7];
      uniform float u_hue[7];
      uniform float u_is_original;

      vec3 srgb_to_linear(vec3 c) {
        return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
      }

      vec3 linear_to_srgb(vec3 c) {
        c = max(c, vec3(0.0));
        return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
      }

      vec3 linear_srgb_to_oklab(vec3 c) {
        float l = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b;
        float m = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b;
        float s = 0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b;

        float l_ = pow(max(l, 0.0), 1.0 / 3.0);
        float m_ = pow(max(m, 0.0), 1.0 / 3.0);
        float s_ = pow(max(s, 0.0), 1.0 / 3.0);

        return vec3(
          0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
          1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
          0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
        );
      }

      vec3 oklab_to_linear_srgb(vec3 c) {
        float l_ = c.x + 0.3963377774 * c.y + 0.2158037573 * c.z;
        float m_ = c.x - 0.1055613458 * c.y - 0.0638541728 * c.z;
        float s_ = c.x - 0.0894841775 * c.y - 1.2914855480 * c.z;

        float l = l_ * l_ * l_;
        float m = m_ * m_ * m_;
        float s = s_ * s_ * s_;

        return vec3(
          +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
          -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
          -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
        );
      }

      vec3 oklab_to_oklch(vec3 oklab) {
        float l = oklab.x;
        float c = sqrt(oklab.y * oklab.y + oklab.z * oklab.z);
        float h = atan(oklab.z, oklab.y) * 57.29577951308232;
        if (h < 0.0) h += 360.0;
        return vec3(l, c, h);
      }

      vec3 oklch_to_oklab(vec3 oklch) {
        float h_rad = oklch.z * 0.017453292519943295;
        return vec3(
          oklch.x,
          oklch.y * cos(h_rad),
          oklch.y * sin(h_rad)
        );
      }

      float eval_cyclic_spline(float h_deg, float p0, float p1, float p2, float p3, float p4, float p5, float p6) {
        float h = mod(h_deg, 360.0);
        if (h < 0.0) h += 360.0;
        float norm = h / 60.0;
        int idx = int(floor(norm));
        if (idx > 5) idx = 5;
        float t = norm - float(idx);

        float cp0 = (idx == 0) ? p5 : (idx == 1 ? p0 : (idx == 2 ? p1 : (idx == 3 ? p2 : (idx == 4 ? p3 : p4))));
        float cp1 = (idx == 0) ? p0 : (idx == 1 ? p1 : (idx == 2 ? p2 : (idx == 3 ? p3 : (idx == 4 ? p4 : p5))));
        float cp2 = (idx == 0) ? p1 : (idx == 1 ? p2 : (idx == 2 ? p3 : (idx == 3 ? p4 : (idx == 4 ? p5 : p6))));
        float cp3 = (idx == 0) ? p2 : (idx == 1 ? p3 : (idx == 2 ? p4 : (idx == 3 ? p5 : (idx == 4 ? p6 : p1))));

        float t2 = t * t;
        float t3 = t2 * t;
        return 0.5 * (
          (2.0 * cp1) +
          (-cp0 + cp2) * t +
          (2.0 * cp0 - 5.0 * cp1 + 4.0 * cp2 - cp3) * t2 +
          (-cp0 + 3.0 * cp1 - 3.0 * cp2 + cp3) * t3
        );
      }

      void main() {
        vec4 src = texture2D(u_image, v_uv);
        if (u_is_original > 0.5) {
          gl_FragColor = src;
          return;
        }

        vec3 linear_rgb = srgb_to_linear(src.rgb);
        vec3 oklab = linear_srgb_to_oklab(linear_rgb);
        vec3 oklch = oklab_to_oklch(oklab);

        float protection_mask = smoothstep(0.0, 0.04, oklch.y);

        // 1. Saturation scale
        float sat_scale = eval_cyclic_spline(oklch.z, u_sat[0], u_sat[1], u_sat[2], u_sat[3], u_sat[4], u_sat[5], u_sat[6]);
        oklch.y *= mix(1.0, max(sat_scale, 0.0), protection_mask);

        // 2. Brightness offset
        float bri_offset = eval_cyclic_spline(oklch.z, u_bri[0], u_bri[1], u_bri[2], u_bri[3], u_bri[4], u_bri[5], u_bri[6]);
        oklch.x = clamp(oklch.x + bri_offset * protection_mask, 0.0, 1.0);

        // 3. Hue shift
        float hue_shift = eval_cyclic_spline(oklch.z, u_hue[0], u_hue[1], u_hue[2], u_hue[3], u_hue[4], u_hue[5], u_hue[6]);
        oklch.z = mod(oklch.z + hue_shift * protection_mask, 360.0);
        if (oklch.z < 0.0) oklch.z += 360.0;

        vec3 corrected = oklab_to_linear_srgb(oklch_to_oklab(oklch));
        vec3 result = linear_to_srgb(corrected);

        gl_FragColor = vec4(result, src.a);
      }
    `;

    function createShader(type, src) {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      return s;
    }

    const vs = createShader(gl.VERTEX_SHADER, vsSource);
    const fs = createShader(gl.FRAGMENT_SHADER, fsSource);
    glProgram = gl.createProgram();
    gl.attachShader(glProgram, vs);
    gl.attachShader(glProgram, fs);
    gl.linkProgram(glProgram);
    gl.useProgram(glProgram);

    const quad = new Float32Array([
      -1, -1,  0, 1,
       1, -1,  1, 1,
      -1,  1,  0, 0,
      -1,  1,  0, 0,
       1, -1,  1, 1,
       1,  1,  1, 0,
    ]);

    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);

    const aPos = gl.getAttribLocation(glProgram, 'a_pos');
    const aUv = gl.getAttribLocation(glProgram, 'a_uv');
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 16, 0);
    gl.enableVertexAttribArray(aUv);
    gl.vertexAttribPointer(aUv, 2, gl.FLOAT, false, 16, 8);

    uSatLoc = gl.getUniformLocation(glProgram, 'u_sat');
    uBriLoc = gl.getUniformLocation(glProgram, 'u_bri');
    uHueLoc = gl.getUniformLocation(glProgram, 'u_hue');
    uIsOrigLoc = gl.getUniformLocation(glProgram, 'u_is_original');

    glTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, glTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    function loadSamplePhoto(srcUrl) {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        try {
          gl.bindTexture(gl.TEXTURE_2D, glTexture);
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
          renderWebGL();
        } catch (e) {
          const imgFallback = new Image();
          imgFallback.onload = () => {
            try {
              gl.bindTexture(gl.TEXTURE_2D, glTexture);
              gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, imgFallback);
              renderWebGL();
            } catch (err) {}
          };
          imgFallback.src = srcUrl;
        }
      };
      img.onerror = () => {
        const imgFallback = new Image();
        imgFallback.onload = () => {
          try {
            gl.bindTexture(gl.TEXTURE_2D, glTexture);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, imgFallback);
            renderWebGL();
          } catch (err) {}
        };
        imgFallback.src = srcUrl;
      };
      img.src = srcUrl;
    }

    loadSamplePhoto(window.SAMPLE_LANDSCAPE_B64 || 'assets/images/sample_landscape.jpg');

    const btnSample1 = document.getElementById('btn-color-sample-1');
    const btnSample2 = document.getElementById('btn-color-sample-2');
    if (btnSample1) {
      btnSample1.addEventListener('click', () => {
        btnSample1.classList.add('btn-primary');
        btnSample1.classList.remove('btn-secondary');
        if (btnSample2) {
          btnSample2.classList.add('btn-secondary');
          btnSample2.classList.remove('btn-primary');
        }
        loadSamplePhoto(window.SAMPLE_LANDSCAPE_B64 || 'assets/images/sample_landscape.jpg');
      });
    }
    if (btnSample2) {
      btnSample2.addEventListener('click', () => {
        btnSample2.classList.add('btn-primary');
        btnSample2.classList.remove('btn-secondary');
        if (btnSample1) {
          btnSample1.classList.add('btn-secondary');
          btnSample1.classList.remove('btn-primary');
        }
        loadSamplePhoto(window.SAMPLE_PORTRAIT_B64 || 'assets/images/sample_portrait.jpg');
      });
    }

    const btnCompare = document.getElementById('btn-color-compare');
    if (btnCompare) {
      const showOrig = (e) => { e.preventDefault(); isShowingOriginal = true; renderWebGL(); };
      const hideOrig = (e) => { e.preventDefault(); isShowingOriginal = false; renderWebGL(); };
      btnCompare.addEventListener('mousedown', showOrig);
      btnCompare.addEventListener('touchstart', showOrig, { passive: false });
      btnCompare.addEventListener('mouseup', hideOrig);
      btnCompare.addEventListener('mouseleave', hideOrig);
      btnCompare.addEventListener('touchend', hideOrig);
    }
  }

  function renderWebGL() {
    if (!gl || !glProgram) return;
    gl.viewport(0, 0, glCanvas.width, glCanvas.height);
    gl.useProgram(glProgram);
    gl.uniform1fv(uSatLoc, new Float32Array(satNodes));
    gl.uniform1fv(uBriLoc, new Float32Array(briNodes));
    gl.uniform1fv(uHueLoc, new Float32Array(hueNodes));
    gl.uniform1f(uIsOrigLoc, isShowingOriginal ? 1.0 : 0.0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  function getActivePointsArray() {
    if (currentMode === 'saturation') return satNodes;
    if (currentMode === 'brightness') return briNodes;
    return hueNodes;
  }

  function valueToY(val) {
    const scale = (midY - padY);
    if (currentMode === 'saturation') {
      return midY - (val - 1.0) * scale;
    }
    if (currentMode === 'brightness') {
      return midY - (val / 0.35) * scale;
    }
    return midY - (val / 60.0) * scale;
  }

  function yToValue(y) {
    const scale = (midY - padY);
    const norm = (midY - y) / scale;
    if (currentMode === 'saturation') {
      return Math.max(0.0, Math.min(2.0, 1.0 + norm));
    }
    if (currentMode === 'brightness') {
      return Math.max(-0.35, Math.min(0.35, norm * 0.35));
    }
    return Math.max(-60.0, Math.min(60.0, norm * 60.0));
  }

  function formatValueBadge(val) {
    if (currentMode === 'saturation') {
      return `${val.toFixed(2)}× Sat`;
    }
    if (currentMode === 'brightness') {
      return (val >= 0 ? `+${val.toFixed(2)}` : `${val.toFixed(2)}`) + ' Bri';
    }
    return (val >= 0 ? `+${Math.round(val)}°` : `${Math.round(val)}°`) + ' Hue';
  }

  function getCanvasCoords(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = w / rect.width;
    const scaleY = h / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    };
  }

  function catmullRom(p0, p1, p2, p3, t) {
    const t2 = t * t;
    const t3 = t2 * t;
    return 0.5 * (
      (2 * p1) +
      (-p0 + p2) * t +
      (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
      (-p0 + 3 * p1 - 3 * p2 + p3) * t3
    );
  }

  function getCurvePoints() {
    const pts = [];
    const points = getActivePointsArray();
    const n = points.length;
    const coords = [];

    for (let i = 0; i < n; i++) {
      coords.push({
        x: padX + i * step,
        y: valueToY(points[i])
      });
    }

    const pVirtualBefore = {
      x: coords[0].x - step,
      y: coords[n - 2].y
    };
    const pVirtualAfter = {
      x: coords[n - 1].x + step,
      y: coords[1].y
    };

    const fullCoords = [pVirtualBefore, ...coords, pVirtualAfter];
    const samplesPerSegment = 24;

    for (let i = 0; i < n - 1; i++) {
      const p0 = fullCoords[i];
      const p1 = fullCoords[i + 1];
      const p2 = fullCoords[i + 2];
      const p3 = fullCoords[i + 3];

      for (let s = 0; s < samplesPerSegment; s++) {
        const t = s / samplesPerSegment;
        pts.push({
          x: catmullRom(p0.x, p1.x, p2.x, p3.x, t),
          y: catmullRom(p0.y, p1.y, p2.y, p3.y, t)
        });
      }
    }
    pts.push(coords[n - 1]);
    return pts;
  }

  function drawSpline() {
    ctx.clearRect(0, 0, w, h);

    // 1. Dark Background Fill
    ctx.fillStyle = '#0a0e14';
    ctx.fillRect(0, 0, w, h);

    // 2. Spectrum Gradient Band at Bottom
    const specGrad = ctx.createLinearGradient(padX, 0, w - padX, 0);
    specGrad.addColorStop(0.000, '#ff3b30');
    specGrad.addColorStop(0.166, '#ffcc00');
    specGrad.addColorStop(0.333, '#34c759');
    specGrad.addColorStop(0.500, '#00c7be');
    specGrad.addColorStop(0.666, '#007aff');
    specGrad.addColorStop(0.833, '#af52de');
    specGrad.addColorStop(1.000, '#ff3b30');
    ctx.fillStyle = specGrad;
    ctx.fillRect(padX - 8, h - 22, (w - 2 * padX) + 16, 5);

    // 3. Zone Columns & Vertical Grid Lines
    const points = getActivePointsArray();
    for (let i = 0; i < points.length; i++) {
      const x = padX + i * step;

      ctx.strokeStyle = (hoveredIdx === i || activeIdx === i) ? 'rgba(0, 210, 255, 0.3)' : 'rgba(255, 255, 255, 0.05)';
      ctx.lineWidth = (hoveredIdx === i || activeIdx === i) ? 1.5 : 1;
      ctx.beginPath();
      ctx.moveTo(x, padY - 10);
      ctx.lineTo(x, h - 26);
      ctx.stroke();

      ctx.fillStyle = (hoveredIdx === i || activeIdx === i) ? '#00d2ff' : '#6e7681';
      ctx.font = (hoveredIdx === i || activeIdx === i) ? '600 10px JetBrains Mono, monospace' : '400 10px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(hueNames[i], x, h - 8);
    }

    // 4. Horizontal Grid & Neutral Baseline
    ctx.strokeStyle = 'rgba(0, 210, 255, 0.4)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(padX - 10, midY);
    ctx.lineTo(w - padX + 10, midY);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = '#00d2ff';
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'right';
    const baseLabel = currentMode === 'saturation' ? '1.0× (100%)' : '0.0';
    ctx.fillText(baseLabel, padX - 14, midY + 3);

    // 5. Spline Fill to Baseline
    const curvePts = getCurvePoints();
    if (curvePts.length > 0) {
      ctx.beginPath();
      ctx.moveTo(curvePts[0].x, midY);
      curvePts.forEach(pt => ctx.lineTo(pt.x, pt.y));
      ctx.lineTo(curvePts[curvePts.length - 1].x, midY);
      ctx.closePath();

      const grad = ctx.createLinearGradient(0, padY, 0, h - padY);
      grad.addColorStop(0, 'rgba(0, 210, 255, 0.25)');
      grad.addColorStop(0.5, 'rgba(0, 210, 255, 0.05)');
      grad.addColorStop(1, 'rgba(0, 210, 255, 0.2)');
      ctx.fillStyle = grad;
      ctx.fill();

      // 6. Glowing Spline Line
      ctx.beginPath();
      curvePts.forEach((pt, idx) => {
        if (idx === 0) ctx.moveTo(pt.x, pt.y);
        else ctx.lineTo(pt.x, pt.y);
      });
      ctx.lineWidth = 3;
      ctx.strokeStyle = currentMode === 'saturation' ? '#38bdf8' : (currentMode === 'brightness' ? '#f59e0b' : '#a855f7');
      ctx.shadowColor = ctx.strokeStyle;
      ctx.shadowBlur = 12;
      ctx.stroke();
      ctx.shadowBlur = 0;
    }

    // 7. Control Point Knobs & Tooltip Badges
    for (let i = 0; i < points.length; i++) {
      const px = padX + i * step;
      const py = valueToY(points[i]);
      const isPointActive = (activeIdx === i);
      const isPointHovered = (hoveredIdx === i);

      if (isPointActive || isPointHovered) {
        ctx.beginPath();
        ctx.arc(px, py, 12, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0, 210, 255, 0.25)';
        ctx.fill();
      }

      ctx.beginPath();
      ctx.arc(px, py, isPointActive ? 7 : (isPointHovered ? 6 : 5), 0, Math.PI * 2);
      ctx.fillStyle = isPointActive ? '#00d2ff' : (isPointHovered ? '#38bdf8' : '#f59e0b');
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#ffffff';
      ctx.stroke();

      const isModified = currentMode === 'saturation' ? Math.abs(points[i] - 1.0) > 0.05 : Math.abs(points[i]) > 0.03;
      if (isPointActive || isPointHovered || isModified) {
        const text = formatValueBadge(points[i]);
        ctx.font = '600 10px JetBrains Mono, monospace';
        const tw = ctx.measureText(text).width;
        const badgeW = tw + 10;
        const badgeH = 18;
        const badgeY = py < midY ? py - 22 : py + 14;

        ctx.fillStyle = isPointActive ? 'rgba(0, 210, 255, 0.9)' : 'rgba(21, 27, 35, 0.9)';
        ctx.beginPath();
        ctx.roundRect(px - badgeW / 2, badgeY - badgeH / 2, badgeW, badgeH, 4);
        ctx.fill();
        ctx.strokeStyle = isPointActive ? '#00d2ff' : 'rgba(255, 255, 255, 0.2)';
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.fillStyle = isPointActive ? '#05080e' : '#e6edf3';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, px, badgeY);
        ctx.textBaseline = 'alphabetic';
      }
    }

    renderWebGL();
  }

  drawSpline();

  function findClosestIndex(canvasX) {
    let bestIdx = 0;
    let minDiff = Infinity;
    const points = getActivePointsArray();
    for (let i = 0; i < points.length; i++) {
      const px = padX + i * step;
      const diff = Math.abs(canvasX - px);
      if (diff < minDiff) {
        minDiff = diff;
        bestIdx = i;
      }
    }
    return bestIdx;
  }

  // Pointer Down (Mouse, Touch, Pen)
  canvas.addEventListener('pointerdown', (e) => {
    const coords = getCanvasCoords(e);
    const idx = findClosestIndex(coords.x);
    activeIdx = idx;
    isDragging = true;
    canvas.setPointerCapture(e.pointerId);

    const points = getActivePointsArray();
    const newVal = yToValue(coords.y);
    points[activeIdx] = newVal;
    if (activeIdx === 0) points[6] = newVal;
    if (activeIdx === 6) points[0] = newVal;
    drawSpline();
  });

  // Pointer Move
  canvas.addEventListener('pointermove', (e) => {
    const coords = getCanvasCoords(e);
    const points = getActivePointsArray();
    if (isDragging && activeIdx !== null) {
      const newVal = yToValue(coords.y);
      points[activeIdx] = newVal;
      if (activeIdx === 0) points[6] = newVal;
      if (activeIdx === 6) points[0] = newVal;
      drawSpline();
    } else {
      const prevHover = hoveredIdx;
      const idx = findClosestIndex(coords.x);
      const px = padX + idx * step;
      if (Math.abs(coords.x - px) <= step * 0.55) {
        hoveredIdx = idx;
        canvas.style.cursor = 'ns-resize';
      } else {
        hoveredIdx = null;
        canvas.style.cursor = 'crosshair';
      }
      if (prevHover !== hoveredIdx) {
        drawSpline();
      }
    }
  });

  function handlePointerEnd(e) {
    if (isDragging) {
      isDragging = false;
      activeIdx = null;
      try {
        canvas.releasePointerCapture(e.pointerId);
      } catch (err) {}
      drawSpline();
    }
  }

  canvas.addEventListener('pointerup', handlePointerEnd);
  canvas.addEventListener('pointercancel', handlePointerEnd);

  // Double click to reset point / curve
  canvas.addEventListener('dblclick', (e) => {
    const coords = getCanvasCoords(e);
    const idx = findClosestIndex(coords.x);
    const px = padX + idx * step;
    const points = getActivePointsArray();
    const defaultVal = currentMode === 'saturation' ? 1.0 : 0.0;
    if (Math.abs(coords.x - px) <= step * 0.5) {
      points[idx] = defaultVal;
      if (idx === 0) points[6] = defaultVal;
      if (idx === 6) points[0] = defaultVal;
    } else {
      for (let i = 0; i < points.length; i++) points[i] = defaultVal;
    }
    drawSpline();
  });

  // 3 Mode Tabs (Saturation / Brightness / Hue)
  const tabGroup = document.getElementById('color-eq-tabs');
  const activePresetLabel = document.getElementById('color-active-preset-name');
  const tipLabel = document.getElementById('color-mode-tip');

  if (tabGroup) {
    tabGroup.addEventListener('click', (e) => {
      const btn = e.target.closest('button[data-mode]');
      if (!btn) return;
      currentMode = btn.getAttribute('data-mode');

      tabGroup.querySelectorAll('button').forEach(b => {
        b.classList.remove('btn-primary');
        b.classList.add('btn-secondary');
      });
      btn.classList.add('btn-primary');
      btn.classList.remove('btn-secondary');

      if (currentMode === 'saturation') {
        if (activePresetLabel) activePresetLabel.textContent = 'Saturation Mode';
        if (tipLabel) tipLabel.innerHTML = '💡 <strong>Saturation Mode</strong>: Drag points to selectively boost or desaturate specific hues across the spectrum.';
      } else if (currentMode === 'brightness') {
        if (activePresetLabel) activePresetLabel.textContent = 'Brightness Mode';
        if (tipLabel) tipLabel.innerHTML = '💡 <strong>Brightness Mode</strong>: Darken or brighten specific color ranges (e.g. deepen blue skies or lift skin tones).';
      } else if (currentMode === 'hue') {
        if (activePresetLabel) activePresetLabel.textContent = 'Hue Shift Mode';
        if (tipLabel) tipLabel.innerHTML = '💡 <strong>Hue Shift Mode</strong>: Shift color spectrums seamlessly (e.g. turn summer greens into golden autumn foliage).';
      }

      drawSpline();
    });
  }

  // Presets
  const presetsContainer = document.getElementById('color-spline-presets');
  if (presetsContainer) {
    presetsContainer.addEventListener('click', (e) => {
      const btn = e.target.closest('button[data-preset]');
      if (!btn) return;
      const preset = btn.getAttribute('data-preset');

      if (preset === 'reset') {
        satNodes.fill(1.0);
        briNodes.fill(0.0);
        hueNodes.fill(0.0);
        if (activePresetLabel) activePresetLabel.textContent = 'Reset (Neutral)';
      } else if (preset === 'mute-greens') {
        satNodes[2] = 0.15; // Green
        satNodes[3] = 0.35; // Cyan
        if (activePresetLabel) activePresetLabel.textContent = 'Mute Greens';
      } else if (preset === 'autumn-foliage') {
        hueNodes[2] = -45.0; // Green -> Yellow/Orange
        satNodes[1] = 1.45;  // Boost Yellow
        satNodes[2] = 1.35;  // Boost Green/Autumn
        if (activePresetLabel) activePresetLabel.textContent = 'Autumn Foliage';
      } else if (preset === 'deep-sky') {
        briNodes[4] = -0.28; // Blue
        briNodes[3] = -0.15; // Cyan
        satNodes[4] = 1.30;  // Blue
        if (activePresetLabel) activePresetLabel.textContent = 'Deep Blue Sky';
      } else if (preset === 'teal-orange') {
        hueNodes[4] = -35.0; // Blue -> Teal
        hueNodes[3] = -25.0; // Cyan -> Deep Teal
        hueNodes[0] = +20.0; // Red -> Orange
        satNodes[0] = 1.35;
        satNodes[4] = 1.40;
        if (activePresetLabel) activePresetLabel.textContent = 'Teal & Orange Film Look';
      }

      drawSpline();
    });
  }
}

/* --------------------------------------------------------------------------
   5. Copy Command to Clipboard
   -------------------------------------------------------------------------- */
function initCopyButtons() {
  const copyBtns = document.querySelectorAll('.copy-btn');
  copyBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-target');
      const textElement = targetId ? document.getElementById(targetId) : btn.previousElementSibling;
      if (!textElement) return;

      const text = textElement.textContent.trim();
      navigator.clipboard.writeText(text).then(() => {
        const origText = btn.textContent;
        btn.textContent = '✓ Copied!';
        btn.style.background = '#10b981';
        btn.style.color = '#fff';
        setTimeout(() => {
          btn.textContent = origText;
          btn.style.background = '';
          btn.style.color = '';
        }, 2000);
      });
    });
  });
}

/* --------------------------------------------------------------------------
   6. Documentation Search Filter
   -------------------------------------------------------------------------- */
function initDocsSearch() {
  const searchInput = document.getElementById('docs-search');
  if (!searchInput) return;

  const docNavItems = document.querySelectorAll('.docs-nav-item');
  const docHeadings = document.querySelectorAll('.docs-content h2, .docs-content h3');

  searchInput.addEventListener('input', (e) => {
    const query = e.target.value.toLowerCase();
    docNavItems.forEach(item => {
      const text = item.textContent.toLowerCase();
      item.style.display = text.includes(query) ? 'block' : 'none';
    });
  });
}

/* --------------------------------------------------------------------------
   7. Keyboard Shortcuts Filter
   -------------------------------------------------------------------------- */
function initShortcutsFilter() {
  const filterInput = document.getElementById('shortcuts-search');
  const shortcutRows = document.querySelectorAll('.shortcut-table-row');

  if (!filterInput || !shortcutRows.length) return;

  filterInput.addEventListener('input', (e) => {
    const query = e.target.value.toLowerCase();
    shortcutRows.forEach(row => {
      const text = row.textContent.toLowerCase();
      row.style.display = text.includes(query) ? '' : 'none';
    });
  });
}
