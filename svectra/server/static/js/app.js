// SvectraDB Web Dashboard Client Application
document.addEventListener("DOMContentLoaded", () => {
  let activeCollection = null;
  let samplePoints = [];
  let hoveredPoint = null;

  // UI Elements
  const collectionsListEl = document.getElementById("collectionsList");
  const statVectorsEl = document.getElementById("statTotalVectors");
  const statCollsEl = document.getElementById("statTotalCollections");
  const statLatencyEl = document.getElementById("statAvgLatency");
  const statMemoryEl = document.getElementById("statMemory");

  const collNameBadge = document.getElementById("selectedCollName");
  const collDimBadge = document.getElementById("selectedCollDim");
  const collMetricBadge = document.getElementById("selectedCollMetric");
  const collIndexBadge = document.getElementById("selectedCollIndex");
  const collCountBadge = document.getElementById("selectedCollCount");

  const queryInput = document.getElementById("queryInput");
  const topKInput = document.getElementById("topKInput");
  const topKVal = document.getElementById("topKVal");
  const filterInput = document.getElementById("filterInput");
  const btnRunQuery = document.getElementById("btnRunQuery");
  const queryResultsEl = document.getElementById("queryResults");
  const queryLatencyBadge = document.getElementById("queryLatencyBadge");

  const vectorCanvas = document.getElementById("vectorCanvas");
  let ctx = vectorCanvas ? vectorCanvas.getContext("2d") : null;

  // Tabs
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      document.getElementById(targetId).classList.add("active");

      if (targetId === "tabVisualizer") {
        resizeCanvas();
        loadCanvasPoints();
      }
    });
  });

  // TopK Slider
  if (topKInput) {
    topKInput.addEventListener("input", (e) => {
      topKVal.textContent = e.target.value;
    });
  }

  // Modals
  const createModal = document.getElementById("createModal");
  document.getElementById("btnOpenCreateModal")?.addEventListener("click", () => {
    createModal.classList.add("active");
  });
  document.getElementById("btnCancelCreate")?.addEventListener("click", () => {
    createModal.classList.remove("active");
  });

  // Fetch Telemetry & Metrics
  async function loadMetrics() {
    try {
      const res = await fetch("/api/v1/metrics");
      if (res.ok) {
        const data = await res.json();
        statVectorsEl.textContent = data.total_vectors.toLocaleString();
        statCollsEl.textContent = data.total_collections;
        statLatencyEl.textContent = `${data.average_query_latency_ms} ms`;
        statMemoryEl.textContent = `${data.total_memory_mb} MB`;
      }
    } catch (e) {
      console.warn("Metrics load error:", e);
    }
  }

  // Fetch Collections List
  async function loadCollections(preferredName = null) {
    try {
      const res = await fetch("/api/v1/collections");
      if (res.ok) {
        const data = await res.json();
        const names = data.collections || [];

        collectionsListEl.innerHTML = "";
        if (names.length === 0) {
          collectionsListEl.innerHTML = `<span style="color: var(--text-muted); font-size: 0.85rem;">No collections found. Click "Seed Demo Data" or "New Collection".</span>`;
          activeCollection = null;
          updateCollectionDetails(null);
          return;
        }

        names.forEach((name) => {
          const pill = document.createElement("button");
          pill.className = `pill ${name === (preferredName || activeCollection || names[0]) ? "active" : ""}`;
          pill.textContent = name;
          pill.addEventListener("click", () => {
            selectCollection(name);
          });
          collectionsListEl.appendChild(pill);
        });

        const target = preferredName || activeCollection || names[0];
        selectCollection(target);
      }
    } catch (e) {
      console.error("Collections fetch error:", e);
    }
  }

  // Select a Collection
  async function selectCollection(name) {
    activeCollection = name;
    document.querySelectorAll(".collection-pills .pill").forEach((p) => {
      p.classList.toggle("active", p.textContent === name);
    });

    try {
      const res = await fetch(`/api/v1/collections/${name}`);
      if (res.ok) {
        const stats = await res.json();
        updateCollectionDetails(stats);
        loadCanvasPoints();
      }
    } catch (e) {
      console.error("Failed to load collection details:", e);
    }
  }

  function updateCollectionDetails(stats) {
    if (!stats) {
      collNameBadge.textContent = "None";
      collDimBadge.textContent = "-";
      collMetricBadge.textContent = "-";
      collIndexBadge.textContent = "-";
      collCountBadge.textContent = "0";
      return;
    }
    collNameBadge.textContent = stats.name;
    collDimBadge.textContent = `${stats.dimension} Dim`;
    collMetricBadge.textContent = stats.metric.toUpperCase();
    collIndexBadge.textContent = stats.index_type.toUpperCase();
    collCountBadge.textContent = `${stats.count.toLocaleString()} vectors`;

    // Update SDK snippet
    const codeSnippet = document.getElementById("pythonSnippet");
    if (codeSnippet) {
      codeSnippet.textContent = `import svectradb

client = svectradb.Client(url="http://localhost:8080")
collection = client.get_collection("${stats.name}")

# Vector or Text similarity search with rich metadata filtering
results = collection.query(
    query_text="semantic search engine",
    top_k=5,
    filter={"category": {"$eq": "technology"}}
)

for r in results:
    print(f"[{r.score:.4f}] {r.id}: {r.document}")`;
    }
  }

  // Create Collection Form Submit
  document.getElementById("createCollectionForm")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const name = document.getElementById("newCollName").value.trim();
    const dimension = parseInt(document.getElementById("newCollDim").value);
    const metric = document.getElementById("newCollMetric").value;
    const index_type = document.getElementById("newCollIndex").value;

    try {
      const res = await fetch("/api/v1/collections", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, dimension, metric, index_type }),
      });
      if (res.ok) {
        createModal.classList.remove("active");
        await loadCollections(name);
        await loadMetrics();
      } else {
        const err = await res.json();
        alert(`Error: ${err.detail || "Failed to create collection"}`);
      }
    } catch (err) {
      alert(`Network error: ${err.message}`);
    }
  });

  // Seed Demo Data
  document.getElementById("btnSeedData")?.addEventListener("click", async () => {
    const btn = document.getElementById("btnSeedData");
    btn.disabled = true;
    btn.textContent = "Seeding...";

    try {
      // 1. Create AI Research collection
      await fetch("/api/v1/collections", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: "ai_papers",
          dimension: 128,
          metric: "cosine",
          index_type: "hnsw",
          index_params: { M: 16, ef_construction: 100 },
        }),
      });

      // 2. Upsert demo vectors + text + metadata
      const demoDocs = [
        {
          id: "paper_01",
          doc: "Attention Is All You Need: The Transformer architecture relies solely on self-attention mechanisms without recurrence.",
          meta: { category: "deep_learning", year: 2017, citations: 95000 },
        },
        {
          id: "paper_02",
          doc: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding using masked language modeling.",
          meta: { category: "nlp", year: 2018, citations: 75000 },
        },
        {
          id: "paper_03",
          doc: "Deep Residual Learning for Image Recognition: ResNet solves vanishing gradient problems using skip connections.",
          meta: { category: "computer_vision", year: 2016, citations: 160000 },
        },
        {
          id: "paper_04",
          doc: "Hierarchical Navigable Small World (HNSW): Logarithmic scale approximate nearest neighbor vector search graphs.",
          meta: { category: "vector_db", year: 2020, citations: 3200 },
        },
        {
          id: "paper_05",
          doc: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks: Combining dense vector retrieval with LLM generation.",
          meta: { category: "nlp", year: 2020, citations: 14000 },
        },
        {
          id: "paper_06",
          doc: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.",
          meta: { category: "systems", year: 2022, citations: 4500 },
        },
        {
          id: "paper_07",
          doc: "SvectraDB: High Performance Vector Engine with SIMD Vector Math and Durability.",
          meta: { category: "vector_db", year: 2026, citations: 10 },
        },
      ];

      await fetch("/api/v1/collections/ai_papers/upsert", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ids: demoDocs.map((d) => d.id),
          documents: demoDocs.map((d) => d.doc),
          metadatas: demoDocs.map((d) => d.meta),
        }),
      });

      await loadCollections("ai_papers");
      await loadMetrics();
    } catch (e) {
      console.error("Failed to seed data:", e);
    } finally {
      btn.disabled = false;
      btn.textContent = "⚡ Seed Demo Data";
    }
  });

  // Run Similarity Search
  btnRunQuery?.addEventListener("click", async () => {
    if (!activeCollection) {
      alert("Please select or create a collection first.");
      return;
    }

    const qText = queryInput.value.trim();
    const topK = parseInt(topKInput.value);
    let filterObj = null;

    if (filterInput.value.trim()) {
      try {
        filterObj = JSON.parse(filterInput.value.trim());
      } catch (e) {
        alert("Invalid JSON in filter: " + e.message);
        return;
      }
    }

    btnRunQuery.disabled = true;
    btnRunQuery.textContent = "Searching...";

    try {
      const res = await fetch(`/api/v1/collections/${activeCollection}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query_text: qText || "deep learning transformers",
          top_k: topK,
          filter: filterObj,
          include_document: true,
          include_metadata: true,
        }),
      });

      if (res.ok) {
        const data = await res.json();
        queryLatencyBadge.textContent = `${data.latency_ms} ms`;
        renderResults(data.results);
        await loadMetrics();
      } else {
        const err = await res.json();
        alert("Query error: " + (err.detail || "Failed"));
      }
    } catch (e) {
      alert("Network error: " + e.message);
    } finally {
      btnRunQuery.disabled = false;
      btnRunQuery.textContent = "Run Similarity Search";
    }
  });

  function renderResults(results) {
    queryResultsEl.innerHTML = "";
    if (!results || results.length === 0) {
      queryResultsEl.innerHTML = `<div style="color: var(--text-muted); font-size: 0.9rem; padding: 1rem;">No matching vectors found.</div>`;
      return;
    }

    results.forEach((r, idx) => {
      const card = document.createElement("div");
      card.className = "result-card";
      card.innerHTML = `
        <div class="result-header">
          <span class="result-id">#${idx + 1} • ${r.id}</span>
          <span class="score-badge">Score: ${(r.score * 100).toFixed(1)}%</span>
        </div>
        <div class="result-text">${r.document || "No raw document attached"}</div>
        <div class="result-meta">Metadata: ${JSON.stringify(r.metadata || {})}</div>
      `;
      queryResultsEl.appendChild(card);
    });
  }

  // 2D Canvas Visualization
  function resizeCanvas() {
    if (!vectorCanvas) return;
    const rect = vectorCanvas.parentElement.getBoundingClientRect();
    vectorCanvas.width = rect.width * window.devicePixelRatio;
    vectorCanvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
  }

  async function loadCanvasPoints() {
    if (!activeCollection || !vectorCanvas) return;
    try {
      const res = await fetch(`/api/v1/collections/${activeCollection}/sample?limit=150`);
      if (res.ok) {
        const data = await res.json();
        samplePoints = data.points || [];
        drawCanvas();
      }
    } catch (e) {
      console.warn("Failed to load sample points:", e);
    }
  }

  function drawCanvas() {
    if (!ctx || !vectorCanvas) return;
    const w = vectorCanvas.clientWidth;
    const h = vectorCanvas.clientHeight;

    ctx.clearRect(0, 0, w, h);

    // Draw grid background
    ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
    ctx.lineWidth = 1;
    const step = 40;
    for (let x = 0; x < w; x += step) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let y = 0; y < h; y += step) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    if (samplePoints.length === 0) {
      ctx.fillStyle = "#64748b";
      ctx.font = "14px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("No vectors in this collection to visualize yet.", w / 2, h / 2);
      return;
    }

    // Determine min/max for normalization
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    samplePoints.forEach((p) => {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    });

    const spanX = maxX - minX || 1;
    const spanY = maxY - minY || 1;
    const pad = 60;

    // Draw connecting edges
    ctx.strokeStyle = "rgba(0, 242, 254, 0.12)";
    for (let i = 0; i < samplePoints.length; i++) {
      for (let j = i + 1; j < samplePoints.length; j++) {
        const p1 = samplePoints[i];
        const p2 = samplePoints[j];
        const d = Math.hypot(p1.x - p2.x, p1.y - p2.y);
        if (d < (spanX + spanY) * 0.15) {
          const x1 = pad + ((p1.x - minX) / spanX) * (w - pad * 2);
          const y1 = pad + ((p1.y - minY) / spanY) * (h - pad * 2);
          const x2 = pad + ((p2.x - minX) / spanX) * (w - pad * 2);
          const y2 = pad + ((p2.y - minY) / spanY) * (h - pad * 2);
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
          ctx.stroke();
        }
      }
    }

    // Draw vector nodes
    samplePoints.forEach((p) => {
      const cx = pad + ((p.x - minX) / spanX) * (w - pad * 2);
      const cy = pad + ((p.y - minY) / spanY) * (h - pad * 2);
      p._cx = cx;
      p._cy = cy;

      const isHovered = hoveredPoint && hoveredPoint.id === p.id;
      const radius = isHovered ? 8 : 5;

      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);

      if (isHovered) {
        ctx.fillStyle = "#ffffff";
        ctx.shadowColor = "#00f2fe";
        ctx.shadowBlur = 15;
      } else {
        ctx.fillStyle = "#00f2fe";
        ctx.shadowColor = "#7000ff";
        ctx.shadowBlur = 8;
      }

      ctx.fill();
      ctx.restore();

      // Tooltip for hovered point
      if (isHovered) {
        ctx.fillStyle = "rgba(10, 14, 26, 0.95)";
        ctx.strokeStyle = "rgba(0, 242, 254, 0.5)";
        ctx.lineWidth = 1;
        const ttW = 240;
        const ttH = 65;
        let ttX = cx + 15;
        let ttY = cy - 20;
        if (ttX + ttW > w) ttX = cx - ttW - 15;

        ctx.fillRect(ttX, ttY, ttW, ttH);
        ctx.strokeRect(ttX, ttY, ttW, ttH);

        ctx.fillStyle = "#00f2fe";
        ctx.font = "bold 12px JetBrains Mono, monospace";
        ctx.textAlign = "left";
        ctx.fillText(p.id, ttX + 10, ttY + 20);

        ctx.fillStyle = "#94a3b8";
        ctx.font = "11px Inter, sans-serif";
        const docSnippet = (p.document || "No document").substring(0, 35) + "...";
        ctx.fillText(docSnippet, ttX + 10, ttY + 40);
        ctx.fillText(`Pos: (${p.x.toFixed(2)}, ${p.y.toFixed(2)})`, ttX + 10, ttY + 55);
      }
    });
  }

  if (vectorCanvas) {
    vectorCanvas.addEventListener("mousemove", (e) => {
      const rect = vectorCanvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;

      let found = null;
      for (const p of samplePoints) {
        if (p._cx && p._cy) {
          if (Math.hypot(mx - p._cx, my - p._cy) < 12) {
            found = p;
            break;
          }
        }
      }

      if (found !== hoveredPoint) {
        hoveredPoint = found;
        drawCanvas();
      }
    });
  }

  window.addEventListener("resize", () => {
    resizeCanvas();
    drawCanvas();
  });

  // Initial Boot
  loadMetrics();
  loadCollections();
  setInterval(loadMetrics, 5000);
});
