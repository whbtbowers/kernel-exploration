var diagram = flowchart.parse(
"st=>start: Value of 𝛾 selected|past:>www.google.com\ne=>end: End|future:>www.google.com\nop1=>operation: My Operation|past\nop2=>operation: Stuff|current\nsub1=>subroutine: My Subroutine|invalid\ncond=>condition: Yes\nor No?|approved:>http:www.google.com\nc2=>condition: Good idea|rejected\nio=>inputoutput: catch something...|future\n\nst->op1(right)->cond\ncond(yes, right)->c2\ncond(no)->sub1(left)->op1\nc2(yes)->io->e\nc2(no)->op2->e"
);

diagram.drawSVG('diagram')//, {
//                             'x': 0,
//                             'y': 0,
//                             'line-width': 3,
//                             'line-length': 50,
//                             'text-margin': 10,
//                             'font-size': 14,
//                             'font-color': 'black',
//                             'line-color': 'black',
//                             'element-color': 'black',
//                             'fill': 'white',
//                             'yes-text': 'yes',
//                             'no-text': 'no',
//                             'arrow-end': 'block',
//                             'scale': 1,
//                             // style symbol types
//                             'symbols': {
//                               'start': {
//                                 'font-color': 'red',
//                                 'element-color': 'green',
//                                 'fill': 'yellow'
//                               },
//                               'end':{
//                                 'class': 'end-element'
//                               }
//                             },
//                             // even flowstate support ;-)
//                             'flowstate' : {
//                               'past' : { 'fill' : '#CCCCCC', 'font-size' : 12},
//                               'current' : {'fill' : 'yellow', 'font-color' : 'red', 'font-weight' : 'bold'},
//                               'future' : { 'fill' : '#FFFF99'},
//                               'request' : { 'fill' : 'blue'},
//                               'invalid': {'fill' : '#444444'},
//                               'approved' : { 'fill' : '#58C4A3', 'font-size' : 12, 'yes-text' : 'APPROVED', 'no-text' : 'n/a' },
//                               'rejected' : { 'fill' : '#C45879', 'font-size' : 12, 'yes-text' : 'n/a', 'no-text' : 'REJECTED' }
//                             }
//                           });
