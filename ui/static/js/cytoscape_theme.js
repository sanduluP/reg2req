// ui/static/js/cytoscape_theme.js
// Centralizes: Bootstrap theme extraction + Cytoscape style + Cytoscape creation.

function getCssVar(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

export function getBootstrapTheme() {
  // Bootstrap 5 defaults as fallbacks (only used if CSS vars missing)
  const bsWarning = getCssVar("--bs-warning", "#ffc107");

  return {
    canvas: "#f8fafc",
    nodeText: "#172033",
    nodeBorder: "#ffffff",
    edge: "#94a3b8",
    edgeMuted: "#cbd5e1",
    edgeText: "#334155",
    labelBackground: "#ffffff",
    highlight: bsWarning,
    selected: "#2563eb",
    hub: "#0f766e",
    nodePalette: [
      "#2563eb",
      "#0f766e",
      "#7c3aed",
      "#c2410c",
      "#be123c",
      "#047857",
      "#0369a1",
      "#a16207",
    ],
    bsWarning,
  };
}

export function buildCytoscapeStyle(theme) {
  const {
    nodeText,
    nodeBorder,
    edge,
    edgeMuted,
    edgeText,
    labelBackground,
    highlight,
    selected,
    hub,
  } = theme;

  return [
    {
      selector: "node",
      style: {
        "background-color": "data(color)",
        "border-width": 3,
        "border-color": nodeBorder,
        "border-opacity": 1,
        "label": "data(displayLabel)",
        "text-valign": "bottom",
        "text-halign": "center",
        "text-margin-y": 8,
        "color": nodeText,
        "text-background-opacity": 0.94,
        "text-background-color": labelBackground,
        "text-background-padding": 4,
        "text-background-shape": "roundrectangle",
        "text-wrap": "wrap",
        "text-max-width": 112,
        "min-zoomed-font-size": 8,
        "font-size": "11px",
        "font-weight": 600,
        "width": "data(size)",
        "height": "data(size)",
        "transition-property": "background-color, border-color, border-width, width, height",
        "transition-duration": "0.18s",
      },
    },
    {
      selector: "node.hub-node",
      style: {
        "background-color": hub,
        "width": "data(hubSize)",
        "height": "data(hubSize)",
      },
    },
    {
      selector: "node.hovered-node",
      style: {
        "border-width": 5,
        "border-color": selected,
      },
    },
    {
      selector: "edge",
      style: {
        "width": 1.6,
        "line-color": edgeMuted,
        "target-arrow-color": edgeMuted,
        "target-arrow-shape": "triangle",
        "curve-style": "bezier",
        "control-point-step-size": 48,
        "arrow-scale": 1.05,
        "label": "",
        "font-size": "10px",
        "font-weight": 600,
        "color": edgeText,
        "text-rotation": "autorotate",
        "text-background-opacity": 1,
        "text-background-color": labelBackground,
        "text-background-padding": 3,
        "text-background-shape": "roundrectangle",
        "text-wrap": "wrap",
        "text-max-width": 116,
        "min-zoomed-font-size": 8,
        "transition-property": "line-color, target-arrow-color, width",
        "transition-duration": "0.18s",
      },
    },

    {
      selector: ".highlighted-node",
      style: {
        "border-width": 7,
        "border-color": highlight,
        "border-opacity": 1,
        "transition-property": "border-width, border-opacity",
        "transition-duration": "0.2s",
      },
    },

    {
      selector: ".highlighted-edge",
      style: {
        "width": 4.5,
        "line-color": highlight,
        "target-arrow-color": highlight,
        "label": "data(displayLabel)",
        "transition-property": "width, line-color, target-arrow-color",
        "transition-duration": "0.2s",
      },
    },
    {
      selector: ".hovered-edge",
      style: {
        "width": 3,
        "line-color": edge,
        "target-arrow-color": edge,
        "label": "data(displayLabel)",
      },
    },

    {
      selector: ".faded",
      style: {
        "opacity": 0.1,
        "text-opacity": 0,
      },
    },
  ];
}

export function buildGraphLayoutOptions(elements = {}, overrides = {}) {
  const nodeCount = elements?.nodes?.length ?? 0;
  const edgeCount = elements?.edges?.length ?? 0;
  const largeGraph = nodeCount > 45 || edgeCount > 80;

  return {
    name: "cose",
    animate: nodeCount <= 70,
    refresh: 20,
    randomize: true,
    fit: false,
    padding: 80,
    componentSpacing: largeGraph ? 160 : 120,
    nodeOverlap: largeGraph ? 28 : 20,
    idealEdgeLength: largeGraph ? 210 : 170,
    nodeRepulsion: largeGraph ? 180000 : 105000,
    edgeElasticity: 90,
    nestingFactor: 1.2,
    gravity: largeGraph ? 0.18 : 0.28,
    numIter: largeGraph ? 2200 : 1500,
    ...overrides,
  };
}

export function createStyledCytoscape(containerEl, layoutOverrides = {}) {
  const theme = getBootstrapTheme();

  const cy = cytoscape({
    container: containerEl,
    style: buildCytoscapeStyle(theme),
    layout: buildGraphLayoutOptions({}, layoutOverrides),
    wheelSensitivity: 0.18,
    minZoom: 0.18,
    maxZoom: 2.5,
  });

  return { cy, theme };
}
