/**
 * Interactive canvas overlay for orthopedic hardware demonstration.
 * Educational visualization only — not for operative planning.
 */
(function (global) {
  function boneAxis(width, height, detections) {
    if (detections && detections.length) {
      const box = detections[0].box || detections[0];
      const [x1, y1, x2, y2] = box;
      return { x1: (x1 + x2) / 2, y1, x2: (x1 + x2) / 2, y2 };
    }
    return { x1: width * 0.5, y1: height * 0.18, x2: width * 0.5, y2: height * 0.82 };
  }

  function drawHardware(ctx, axis, mode) {
    const { x1, y1, x2, y2 } = axis;
    ctx.lineCap = "round";
    if (mode === "rod") {
      ctx.strokeStyle = "rgba(210, 220, 230, 0.92)";
      ctx.lineWidth = 10;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    } else {
      ctx.strokeStyle = "rgba(46, 196, 182, 0.85)";
      ctx.lineWidth = 16;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
      ctx.strokeStyle = "rgba(230, 240, 245, 0.9)";
      ctx.lineWidth = 7;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }

    const n = mode === "screws" ? 8 : 6;
    const dx = x2 - x1;
    const dy = y2 - y1;
    const len = Math.hypot(dx, dy) || 1;
    const nx = -dy / len;
    const ny = dx / len;
    for (let i = 0; i < n; i += 1) {
      const t = 0.08 + (0.84 * i) / Math.max(n - 1, 1);
      const x = x1 + t * dx;
      const y = y1 + t * dy;
      ctx.strokeStyle = "rgba(200, 210, 220, 0.95)";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(x + nx * 22, y + ny * 22);
      ctx.lineTo(x - nx * 22, y - ny * 22);
      ctx.stroke();
      ctx.fillStyle = "#1b2428";
      ctx.beginPath();
      ctx.arc(x, y, 3.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function mount(canvas, config) {
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const image = new Image();
    image.crossOrigin = "anonymous";
    let mode = "plate";
    let axis = boneAxis(canvas.width, canvas.height, config.detections);
    let dragging = false;

    function render() {
      ctx.fillStyle = "#050c11";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      if (image.naturalWidth) {
        const scale = Math.min(canvas.width / image.naturalWidth, canvas.height / image.naturalHeight);
        const w = image.naturalWidth * scale;
        const h = image.naturalHeight * scale;
        const ox = (canvas.width - w) / 2;
        const oy = (canvas.height - h) / 2;
        ctx.drawImage(image, ox, oy, w, h);
      }
      drawHardware(ctx, axis, mode);
      ctx.fillStyle = "rgba(10, 28, 36, 0.85)";
      ctx.fillRect(12, 12, 420, 36);
      ctx.fillStyle = "#d7fff6";
      ctx.font = "14px Segoe UI, sans-serif";
      ctx.fillText(`REFIXATION  |  ${config.predictedClass || ""}  |  ${mode}`, 24, 36);
    }

    image.onload = render;
    image.src = config.sourceUrl || config.imageUrl || "";

    canvas.addEventListener("mousedown", (event) => {
      dragging = true;
      const rect = canvas.getBoundingClientRect();
      const x = ((event.clientX - rect.left) / rect.width) * canvas.width;
      const y = ((event.clientY - rect.top) / rect.height) * canvas.height;
      axis = { x1: x, y1: y, x2: x, y2: y + 180 };
      render();
    });
    canvas.addEventListener("mousemove", (event) => {
      if (!dragging) return;
      const rect = canvas.getBoundingClientRect();
      axis.x2 = ((event.clientX - rect.left) / rect.width) * canvas.width;
      axis.y2 = ((event.clientY - rect.top) / rect.height) * canvas.height;
      render();
    });
    window.addEventListener("mouseup", () => {
      dragging = false;
    });

    document.querySelectorAll("[data-hw]").forEach((btn) => {
      btn.addEventListener("click", () => {
        mode = btn.getAttribute("data-hw");
        render();
      });
    });
    const reset = document.getElementById("resetHw");
    if (reset) {
      reset.addEventListener("click", () => {
        axis = boneAxis(canvas.width, canvas.height, config.detections);
        mode = "plate";
        render();
      });
    }
  }

  global.FractureRefix = { mount };
})(window);
