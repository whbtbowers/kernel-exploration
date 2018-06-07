trace1 = {
  x: ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
  y: [28.8, 28.5, 37.0, 56.8, 69.7, 79.7, 78.5, 77.8, 74.1, 62.6, 45.3, 39.9],
  line: {
    color: 'rgb(205, 12, 24)',
    width: 4
  },
  name: 'High 2014',
  type: 'scatter',
  xsrc: 'whtbowers:69:3747dd',
  ysrc: 'whtbowers:69:422964'
};
trace2 = {
  x: ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
  y: [12.7, 14.3, 18.6, 35.5, 49.9, 58.0, 60.0, 58.6, 51.7, 45.2, 32.2, 29.1],
  line: {
    color: 'rgb(22, 96, 167)',
    width: 4
  },
  name: 'Low 2014',
  type: 'scatter',
  xsrc: 'whtbowers:69:3747dd',
  ysrc: 'whtbowers:69:fe27a8'
};
trace3 = {
  x: ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
  y: [36.5, 26.6, 43.6, 52.3, 71.5, 81.4, 80.5, 82.2, 76.0, 67.3, 46.1, 35.0],
  line: {
    color: 'rgb(205, 12, 24)',
    dash: 'dash',
    width: 4
  },
  name: 'High 2007',
  type: 'scatter',
  xsrc: 'whtbowers:69:3747dd',
  ysrc: 'whtbowers:69:78f372'
};
trace4 = {
  x: ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
  y: [23.6, 14.0, 27.0, 36.8, 47.6, 57.7, 58.9, 61.2, 53.3, 48.5, 31.0, 23.6],
  line: {
    color: 'rgb(22, 96, 167)',
    dash: 'dash',
    width: 4
  },
  name: 'Low 2007',
  type: 'scatter',
  xsrc: 'whtbowers:69:3747dd',
  ysrc: 'whtbowers:69:009cdb'
};
trace5 = {
  x: ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
  y: [32.5, 37.6, 49.9, 53.0, 69.1, 75.4, 76.5, 76.6, 70.7, 60.6, 45.1, 29.3],
  line: {
    color: 'rgb(205, 12, 24)',
    dash: 'dot',
    width: 4
  },
  name: 'High 2000',
  type: 'scatter',
  xsrc: 'whtbowers:69:3747dd',
  ysrc: 'whtbowers:69:47dbb0'
};
trace6 = {
  x: ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
  y: [13.8, 22.3, 32.5, 37.2, 49.9, 56.1, 57.7, 58.3, 51.2, 42.8, 31.6, 15.9],
  line: {
    color: 'rgb(22, 96, 167)',
    dash: 'dot',
    width: 4
  },
  name: 'Low 2000',
  type: 'scatter',
  xsrc: 'whtbowers:69:3747dd',
  ysrc: 'whtbowers:69:83584c'
};
data = [trace1, trace2, trace3, trace4, trace5, trace6];
layout = {
  title: 'Average High and Low Temperatures in New York',
  xaxis: {title: 'Month'},
  yaxis: {title: 'Temperature (degrees F)'}
};
Plotly.plot('plotly-div-1', {
  data: data,
  layout: layout
});
