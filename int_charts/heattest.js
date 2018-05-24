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
  colorscale: [['0.7', 'rgb(158,1,66)'], ['1.4', 'rgb(213,62,79)'], ['2.1', 'rgb(244,109,67)'], ['2.8', 'rgb(253,174,97)'], ['3.5', 'rgb(254,224,139)'], ['4.2', 'rgb(255,255,191)'], ['4.9', 'rgb(230,245,152)'], ['5.6', 'rgb(171,221,164)'], ['6.3', 'rgb(102,194,165)'], ['7.0', 'rgb(50,136,189)'], ['7.7', 'rgb(94,79,162)']],
  type: 'heatmap',
  colorbar:{
      title:'Colorbar label',
      titleside:'right',
  }
};
data = [trace1];
layout = {
  legend: {
    bgcolor: '#F5F6F9',
    font: {color: '#4D5663'}
  },
  paper_bgcolor: '#F5F6F9',
  plot_bgcolor: '#F5F6F9',
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
