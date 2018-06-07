BARTEST = document.getElementById('bartest');

var trace1 = {
  x: ['a', 'b', 'c','d','e','f','g','h','i'],
  y: [7.0, 30.0, 66.0, 150.0, 213.0, 230.0, 159.0, 103.0, 28.0, 14.0],
  marker: {
    color: '#0000FF',
    line: {width: 1.0}
  },
  opacity: 1,
  orientation: 'v',
  type: 'bar',
  xaxis: 'x1',
  yaxis: 'y1'
};
var data = [trace1];



Plotly.plot(BARTEST, data);
