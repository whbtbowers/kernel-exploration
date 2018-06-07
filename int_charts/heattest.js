TESTER = document.getElementById('tester');

trace1 = {
  x: ["cucumber", "tomato", "lettuce", "asparagus", "potato", "wheat", "barley"],
  y: ["Farmer Joe", "Upland Bros.", "Smith Gardening", "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."],
  z: [[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
      [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
      [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
      [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
      [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
      [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
      [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]],
  colorscale: 'YIOrRd',
  type: 'heatmap',
  colorbar:{
      title:'Mean area under ROC curve',
      titleside:'right',
  }
};
data = [trace1];
layout = {
  legend: {
    bgcolor: '#FFFFFF',
    font: {color: '#4D5663'}
  },
  paper_bgcolor: '#FFFFFF',
  plot_bgcolor: '#FFFFFF',
  xaxis1: {
    gridcolor: '#E1E5ED',
    tickfont: {color: '#4D5663'},
    title: '',
    titlefont: {color: '#4D5663'},
    zerolinecolor: '#E1E5ED'
  },
  yaxis1: {
    gridcolor: '#E1E5ED',
    tickfont: {color: '#4D5663'},
    title: '',
    titlefont: {color: '#4D5663'},
    zeroline: false,
    zerolinecolor: '#E1E5ED'
  }
};
Plotly.plot('tester', data, layout);
