var summBar = document.getElementById('summary-bar');

var x = ['Diabetes', 'Sex', 'Carotid Artery Calcification', 'Extreme Carotid Artery Calcification', 'Family History of Diabetes', 'Parental history of CVD below age 65', 'Family history of CVD', 'Blood Pressure Treatment', 'Diabetes Treatment', 'Lipids Treatment', 'Plaque'];
var y = [0.44260461760461756, 0.5472515613335903, 0.48759152022235136, 0.49230912061794413, 0.49706519804713045, 0.4812935630520932, 0.5082374109646837, 0.4770254671999638, 0.4666841491841492, 0.48498142894343704, 0.5116070888525979];

var trace1 = {
	x:x,
	y:y,
	marker: {
		color: 'white',
		line: {
			width: 1.0
		}
	},
	opacity: 1,
	orientation: 'v',
	type: 'bar',
	xaxis: 'x1',
	yaxis: 'y1',
	textfont: {
		color:'white',
	},

};

var layout = {
  paper_bgcolor:'#2e3141',
	plot_bgcolor: '#2e3141',
	textfont: {
		color:'white',
	},
	xaxis: {
		title: 'Outcome used as target vector',
		color:'white',
		tickangle:20,
		tickfont: {
			size:10
		},
	},
	yaxis: {
		title: 'Mean AUC',
		color:'white',
	}
};

var data = [trace1];



Plotly.plot(summBar, data, layout);

var strList = ['dia', "sex", "cac", "e_cac", "fh_dia", "cvd_65", "fh_cvd", "bp_treat", "dia_treat", "lip_treat", "plaque"];

summBar.on('plotly_click', function(data){
	var char = x.indexOf(data.points[0].x);
	var corr = strList[char];
	console.log(corr + '_roc.html');
	//window.open('testplot.html');
});
