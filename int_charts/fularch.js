var $ = go.GraphObject.make;

var myDiagram =
  $(go.Diagram, "fullarch",
    {
      "undoManager.isEnabled": true, // enable Ctrl-Z to undo and Ctrl-Y to redo
      layout: $(go.TreeLayout, // specify a Diagram.layout that arranges trees
        { angle: 90, layerSpacing: 35, })
    });

// the template we defined earlier
myDiagram.nodeTemplate =
  $(go.Node, "Vertical",
    { background: "#44CCFF", },
  $(go.TextBlock, "Default Text",
    { margin: 8, stroke: "white", font: "bold 11px sans-serif" },
    new go.Binding("text", "name"))
  );

// define a Link template that routes orthogonally, with no arrowhead
myDiagram.linkTemplate =
  $(go.Link,
    { routing: go.Link.Normal, corner: 5,},
  $(go.Shape, { strokeWidth: 3, stroke: "white",}), // the link shape
  $(go.Shape, { toArrow: "Standard", stroke: "white", fill: "white" })
  );


var model = $(go.GraphLinksModel);
model.nodeDataArray = [
  // { key: "KPCA with i kernels", isGroup: true},
  // { key: "SVM with j kernels", isGroup: true},
  { key: "1", name:"Input data (X) and target outcomes (y)" },
  { key: "2", parent: "1", name: "Filter X for rows and columns >10% filled\nand median impute remaining missing data"},
  { key: "3", parent: "1", name: "Take shape, mean, and covariance of X;\n use to generate simulated datasetsand targets" },
  // { key: "4", parent: "1", name: "Laplacian PCA" },
  // { key: "5", parent: "1", name: "Cosine PCA" },
  // { key: "6", parent: "1", name: "Sigmoid PCA" },
  // { key: "7", parent: "4", name: "Linear SVM" },
  // { key: "8", parent: "5",name: "RBF SVM" },
  // { key: "9", parent: "6",name: "Sigmoid SVM" },
  // { key: "10", parent: "7", name: "Collect i x j matrix of ROC AUCs" },
];
model.linkDataArray = [
  { from: "1", to: "2" },
  { from: "2", to: "3" },
  // { from: "KPCA with i kernels", to: "SVM with j kernels" },
  // { from: "SVM with j kernels", to: "10" },

];
myDiagram.model = model;
