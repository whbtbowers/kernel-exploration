var $ = go.GraphObject.make;

var myDiagram =
  $(go.Diagram, "myDiagramDiv",
    {
      "undoManager.isEnabled": true, // enable Ctrl-Z to undo and Ctrl-Y to redo
      layout: $(go.TreeLayout, // specify a Diagram.layout that arranges trees
        { angle: 90, layerSpacing: 70, })
    });

// the template we defined earlier
myDiagram.nodeTemplate =
  $(go.Node, "Vertical",
    { background: "#44CCFF", },
  $(go.TextBlock, "Default Text",
    { margin: 8, stroke: "white", font: "bold 16px sans-serif" },
    new go.Binding("text", "name"))
  );

// define a Link template that routes orthogonally, with no arrowhead
myDiagram.linkTemplate =
  $(go.Link,
    { routing: go.Link.Normal, corner: 5,},
  $(go.Shape, { strokeWidth: 3, stroke: "black",}), // the link shape
  $(go.Shape, { toArrow: "Standard", stroke: "black" })
  );


var model = $(go.GraphLinksModel);
model.nodeDataArray = [
  { key: "KPCA with i kernels", isGroup: true},
  { key: "SVM with j kernels", isGroup: true},
  { key: "1", name:"Select gamma value" },
  { key: "2", parent: "1", group: "KPCA with i kernels", name: "Linear KPCA"},
  { key: "3", parent: "1", group: "KPCA with i kernels", name: "RBF PCA" },
  { key: "4", parent: "1", group: "KPCA with i kernels", name: "Laplacian PCA" },
  { key: "5", parent: "1", group: "KPCA with i kernels", name: "Cosine PCA" },
  { key: "6", parent: "1", group: "KPCA with i kernels", name: "Sigmoid PCA" },
  { key: "7", parent: "4", group: "SVM with j kernels", name: "Linear SVM" },
  { key: "8", parent: "5",  group: "SVM with j kernels",name: "RBF SVM" },
  { key: "9", parent: "6",  group: "SVM with j kernels",name: "Sigmoid SVM" },
  { key: "10", parent: "7", name: "Collect i x j matrix of ROC AUCs" },
];
model.linkDataArray = [
  { from: "1", to: "KPCA with i kernels" },
  { from: "KPCA with i kernels", to: "SVM with j kernels" },
  { from: "SVM with j kernels", to: "10" },

];
myDiagram.model = model;
