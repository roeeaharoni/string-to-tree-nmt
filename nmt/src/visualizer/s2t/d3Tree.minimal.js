// Inspired by "D3.js Drag and Drop Zoomable Tree" by Rob Schmuecker <robert.schmuecker@gmail.com>
// https://gist.github.com/robschmuecker/7880033


function d3Tree(treeData) {

    // Misc. variables
    var i = 0;
    var duration = 0;
    var root;
	
    // size of the diagram
	var pageWidth = $(document).width();
    var viewerWidth = pageWidth - (0.05 * pageWidth);
    var viewerHeight = 800;

    // this is not really used, as I pretty much apply my own layout.
    // but the tree-layout is still used for the links.
    var tree = d3.layout.tree()
        .size([viewerWidth-20, viewerHeight-120]);

    // define a d3 diagonal projection for use by the node paths later on.
    var diagonal = d3.svg.diagonal()
        .projection(function(d) {
            return [d.x, d.y];
        });
		
    // Define the zoom function for the zoomable tree
    function zoom() {
        svgGroup.attr("transform", "translate(" + d3.event.translate + ")scale(" + d3.event.scale + ")");
    }

    // define the zoomListener which calls the zoom function on the "zoom" event constrained within the scaleExtents
    var zoomListener = d3.behavior.zoom().scaleExtent([0.1, 3]).on("zoom", zoom);

	// remove the previous svg if there
	d3.select("svg").remove();
	
    // define the baseSvg, attaching a class for styling and the zoomListener
    var baseSvg = d3.select("#tree-container").append("svg")
        .attr("width", viewerWidth)
        .attr("height", viewerHeight)
        .attr("class", "overlay")
        .call(zoomListener)
		.on("dblclick.zoom", null);

    var tree_baseline = null; // will be updated inside
    function update(source) {
        // Compute the new height, function counts total children of root node and sets tree height accordingly.
        // This prevents the layout looking squashed when new nodes are made visible or looking sparse when nodes are removed
        // This makes the layout more consistent.
        var levelHeight = [1];
        var childCount = function(level, n) {
            if (n.children && n.children.length > 0) {
                if (levelHeight.length <= level + 1) levelHeight.push(0);

                levelHeight[level + 1] += n.children.length;
                n.children.forEach(function(d) {
                    childCount(level + 1, d);
                });
            }
        };
		childCount(0, root);
		var maxLevel = levelHeight.length+2;

        // add pre-terminal nodes (these will be transparent)
        function add_preterminals(t) {
            if ('touched' in t) return;
            t.touched = true;
            if ('children' in t) {
                t.children.forEach(add_preterminals)
            } else {
                t.children = [{"name":t.name, "s":t.s,"e":t.e,"sid":t.sid,"eid":t.eid}]
                t.name = "";
                t.eid = -1;
                t.sid = -1;
            }
        }
        add_preterminals(root)

        // Compute the new tree layout.
        var nodes = tree.nodes(root);
        // determine node positions: leafs are spaces nicely, and then each
        // non-terminal is in the middle of its span.
        var nodeStartToX = {}
        var nodeEndToX = {}
        var dummy = baseSvg.append("text");
        var start_pos = 0
        // determine the X positions for the leafs.
        nodes.filter(n=>!('children' in n)).forEach((n,i)=>{
            console.log(n.name,"start pos",start_pos)
            nodeStartToX[n.s] = start_pos;
            start_pos += dummy.text(n.name).node().getComputedTextLength()
            nodeEndToX[n.e] = start_pos;
            start_pos += 10; // spacing
        });
        dummy.remove()
        // Set heights between levels based on maxLevel.
        // determine the y position of non-terms, and based on that also of leafs.
        var lowest_nonterm = 0;
        nodes.filter(d=>'children' in d).forEach(function(d,i) {
                d.y = (d.depth * ((viewerHeight-200)/(maxLevel)));
                if (d.y > lowest_nonterm) lowest_nonterm = d.y;
                s = nodeStartToX[d.s];
                e = nodeEndToX[d.e];
                d.x = (s+e)/2;
            });
        tree_baseline = lowest_nonterm + 10;

        // set actual positions of leafs
        nodes.filter(d=>!('children' in d)).forEach(function(d,i) {
                //d.y = viewerHeight-50; 
                d.y = tree_baseline;
                d.x = nodeStartToX[d.s] + (nodeEndToX[d.e] - nodeStartToX[d.s])/2
        });

        // determine the links based on the nodes.
        links = tree.links(nodes);
        // Update the nodes…
        node = svgGroup.selectAll("g.node")
            .data(nodes, function(d) {
                return d.id || (d.id = ++i);
            });

        // Enter any new nodes at the parent's previous position.
        var nodeEnter = node.enter().append("g")
            .attr("class", "node")
            .attr("transform", function(d) {
                return "translate(" + source.x0 + "," + source.y0 + ")";
            })

        // draw the nodes
		nodeEnter.append("rect")
			.attr('class', 'nodeRect')
			// Size of the rectangle/2
			.attr("x", function(d){return -(d.name.length*5+10)/2})
			.attr("y", -10)
			.attr("width", 0)
			.attr("height", 0)
			.style("fill", function(d) {
			    return d._children ? "lightsteelblue" : "#fff";
		});

        nodeEnter.append("text")
            .attr("y", 0)
            .attr("dy", ".35em")
            .attr('class', 'node')
            .attr("text-anchor", "middle")
            .text(function(d) {
                return d.name;
            })
            .style("fill-opacity", 1);

        // Update the text to reflect whether node has children or not.
        node.select('text')
            .attr("y", 0)
            .attr("text-anchor", "middle")
            .text(function(d) {
                return d.name;
            });

        node.select("rect.nodeRect")
            .attr("width", function(d) {
				// Adjust the size of the square according to the label
                return d.children || d._children ? d.name.length*5+10 : d.name.length*5+10;
            })
            .attr("height", function(d) {
                if (d.name == "") return 0;
                return d.children || d._children ? 20 : 20;
            })
            .style("fill", function(d) {
                return d._children ? "lightsteelblue" : "#fffff";
            });

        // Transition nodes to their new position.
        var nodeUpdate = node.transition()
            .duration(duration)
            .attr("transform", function(d) {
                return "translate(" + d.x + "," + d.y + ")";
            });

        // Fade the text in
        nodeUpdate.select("text")
            .style("fill-opacity", 1);

        // Transition exiting nodes to the parent's new position.
        var nodeExit = node.exit().transition()
            .duration(duration)
            .attr("transform", function(d) {
                return "translate(" + source.x + "," + source.y + ")";
            })
            .remove();

        nodeExit.select("circle")
            .attr("r", 0);

        nodeExit.select("text")
            .style("fill-opacity", 1);

        // Update the links…
        var link = svgGroup.selectAll("path.link")
            .data(links, function(d) {
                return d.target.id;
            });

        // Enter any new links at the parent's previous position.
        link.enter().insert("path", "g")
            .attr("class", "link")
			//TODO MARKERS LOOK TERRIBLE
			// .attr("marker-end", "url(#markerArrow)")
            .attr("d", function(d) {
                var o = {x: source.x, y: source.y};
                return diagonal({source: o,target: o});
            });
			// TODO doesn't work with the transition
			// .attr("d", straightLine);

        // Transition links to their new position.
        link.transition()
            .duration(duration)
            .attr("d", diagonal);
			// TODO doesn't work with the transition
            // .attr("d", straightLine);

        // Transition exiting nodes to the parent's new position.
        link.exit().transition()
            .duration(duration)
            .attr("d", function(d) {
                var o = {x: source.x, y: source.y};
                return diagonal({source: o,target: o});
            })
			// TODO doesn't work with the transition
			// .attr("d", straightLine)
            .remove();

        // Stash the old positions for transition.
        nodes.forEach(function(d) {
            d.x0 = d.x;
            d.y0 = d.y;
        });
        return nodes;
    }

    // Append a group which holds all nodes and which the zoom Listener can act upon.
    var svgGroup = baseSvg.append("g");

    // Define the root
    root = treeData;
    root.x0 = viewerWidth / 2;
    root.y0 = 0;

    // Layout the tree initially and center on the root node.
    nodes = update(root);
    d3.select('g').attr("transform", "translate(0,20)");
    return {svg:svgGroup, nodes:nodes,w:viewerWidth,h:tree_baseline+30};
}

function x_position_sent_words(svg, words) {
        var nodeStartToX = {}
        var nodeEndToX = {}
        var start_pos = 0
        var dummy = svg.append("text").attr("class","node")
        words.forEach((w,i)=>{
            nodeStartToX[i] = start_pos;
            start_pos += dummy.text(w).node().getComputedTextLength()
            nodeEndToX[i] = start_pos;
            start_pos += 10; // spacing
        });
        dummy.remove()
        return words.map((w,i)=>{return {s:nodeStartToX[i], e:nodeEndToX[i], m:(nodeStartToX[i]+nodeEndToX[i])/2}});
}

function d3TreeAlign(treeData, sentData, alignData, bpeData, bpeSourceData, bpeAlignData, only_lex) {
    var res = d3Tree(treeData)
    var baseSvg = res.svg
    var nodes = res.nodes
    //var sent = baseSvg.append("g").data(sentData).enter().append("p").text(d => d)
        console.log(sentData)
    var maxNodeX = Math.max(...nodes.map(d=>d.x))
    var positions = x_position_sent_words(baseSvg, sentData)
//    var sentNodes = sentData.map((w,i)=>{return { text:w, x:positions[i].s, y:res.h+40, m:positions[i].m }})
    var sentNodes = bpeSourceData.map((w,i)=>{return { text:w, x:positions[i].s, y:res.h+50, m:positions[i].m }})
    var last_node_x = sentNodes[sentNodes.length-1].x
    var offset = (maxNodeX - last_node_x) / 2

    // add tree source nodes
    baseSvg.append("g").selectAll("text").data(sentNodes).enter().append("text")
        .text(d => d.text)
        .attr("x", d => d.x + offset)
        .attr("y", d => d.y+10)
        .attr('class', 'node')
    d3.select('g').attr("transform", "translate(0,20)");

    console.log(nodes.length)
    if (only_lex) { alignData = alignData.filter(d=>d.type=="lex") }
    alignData = alignData.filter(d=>d.a>0.2)
    id2nodes = {}
    nodes.map(d=>{id2nodes[d.sid] = d; id2nodes[d.eid]=d;})
    
    var type2color = {
        "lex":d3.scale.linear().domain([0.0,1.0]).range(["white","blue"]),
        "open":d3.scale.linear().domain([0.0,1.0]).range(["white","red"]),
        "close":d3.scale.linear().domain([0.0,1.0]).range(["white","green"]),
    }

    // add tree alignments
    baseSvg.append("g").selectAll("line").data(alignData).enter().append("line")
        .attr("x1",d=>id2nodes[+d.tid].x)
        //.attr("y1",d=>id2nodes[+d.tid].y)
        .attr("y1",d=>id2nodes[+d.tid].y+12)
        .attr("x2",d=>sentNodes[+d.sid].m + offset)
        .attr("y2",d=>sentNodes[+d.sid].y - 5)
        .attr("stroke",d=> type2color[d.type](d.a) )

    // The BPE
    var positions = x_position_sent_words(baseSvg, bpeSourceData)
    //    var sentNodes = bpeSourceData.map((w,i)=>{return { text:w, x:positions[i].s, y:res.h+60, m:positions[i].m }})
    //    var last_node_x = sentNodes[sentNodes.length-1].x
    //    var offset = (maxNodeX - last_node_x) / 2

    // add BPE source nodes
    //    baseSvg.append("g").selectAll("text").data(sentNodes).enter().append("text")
    //        .text(d => d.text)
    //        .attr("x", d => d.x + offset)
    //        .attr("y", d => d.y+10)
    //        .attr('class', 'node')

    d3.select('g').attr("transform", "translate(0,20)");
    var maxNodeX = Math.max(...nodes.map(d=>d.x))
    var bpe_positions = x_position_sent_words(baseSvg, bpeData)
    var bpeNodes = bpeData.map((w,i)=>{return { text:w, x:bpe_positions[i].s, y:res.h+135, m:bpe_positions[i].m }})
    var last_node_x = bpeNodes[bpeNodes.length-1].x
    var offset2 = (maxNodeX - last_node_x) / 2

    // add BPE target nodes
    baseSvg.append("g").selectAll("text").data(bpeNodes).enter().append("text")
        .text(d => d.text)
        .attr("x", d => d.x + offset2)
        .attr("y", d => d.y+10)
        .attr('class', 'node')
    d3.select('g').attr("transform", "translate(0,20)");

    bpeAlignData = bpeAlignData.filter(d=>d.type=="lex")
    bpeAlignData = bpeAlignData.filter(d=>d.a>0.2)
    console.log("bpeNodes",bpeNodes.length)

    // add BPE alignments
    baseSvg.append("g").selectAll("line").data(bpeAlignData).enter().append("line")
        .attr("x1",d=>sentNodes[+d.sid].m + offset)
        //.attr("y1",d=>id2nodes[+d.tid].y)
        .attr("y1",d=>sentNodes[+d.sid].y+15)
        .attr("x2",d=>bpeNodes[+d.tid].m + offset2)
        .attr("y2",d=>bpeNodes[+d.tid].y - 5)
        .attr("stroke",d=> type2color[d.type](d.a) )

}

