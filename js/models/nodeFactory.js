define(["data/position", "util/merge", "params"], function(position, merge, config) {


	//////////////////////////////////////////////
	//											//
	//               Interface					//
	//											//
	//////////////////////////////////////////////

	var nodeFactory = {};



	//////////////////////////////////////////////
	//											//
	//                Events					//
	//											//
	//////////////////////////////////////////////
	
	// There are no events in node for a very good reason: If an event is
	// called it will trigger functions for every single node created. This is
	// most likely not what we want, so think about what you are doing before
	// adding an event here. Instead put an event in the nodes model and handle
	// it there



	//////////////////////////////////////////////
	//											//
	//              Initialize					//
	//											//
	//////////////////////////////////////////////
	
	nodeFactory.new = function(data, index) {

		var n =	{
					// Properties
					domNode:	null,
					links:		new Array(),
					//pos:		initPosition(index) ,
					
					//to be complient with force layout:
					x:			initPosition(index).x,
					y:			initPosition(index).y,
					weight:		1,
					index:		index,

					// Methods
					// isScheduled:	isScheduledFun,
					getAbstract:	getAbstractFun,
					getDate:		getDateFun,
					addLink:		addLinkFun,
					doeslinkExist:  doeslinkExistFun
				}

		return merge(n,data);
	}


	//////////////////////////////////////////////
	//											//
	//                Functions					//
	//											//
	//////////////////////////////////////////////
	
	// A function to add a link to a node
	var addLinkFun = function(targetNode, value) {
		
		// Check this link is not already included:
		// ! Comment this line if we want double link
		if(! this.doeslinkExist(targetNode) ){
			var l = {
				source:		this,
				target:		targetNode,
				value:		value,
				domLink:	null,
			}
			
			// Add the link to the node
			this.links.push(l);
		}
	}

	// Fetch an abstract per ajax
	var getAbstractFun = function(callback) {
		// If we have an abstract already, call the callback
		if (this.abstract != undefined) callback(this.abstract)

		// If not, then fetch abstract from server
		else {
			$.get("ajax.php", { task: "abstract", id: this.id }, function (data) { 
				this.abstract = data;
				if (callback != undefined) callback(data);
			});
		}
	}


	// Get date from node
	var getDateFun = function() {
		var date		= new Date(parseInt(this.date) + (new Date()).getTimezoneOffset()*60000)
		return date;
	}

	
	var doeslinkExistFun = function(target) {
		var targets = this.links.map(function(e) { return e.target});
		//if(this.links != []) console.log(this.links);
		return targets.indexOf(target) != -1;
	}
	
	// Find initial position of the node, else create it.
	var initPosition = function(index) {
		var pos = {};
		if(position[index] == null){
			
			pos.x = config['graph_width']*Math.random();
			pos.y = config['graph_height']*Math.random();
			
		}else {
			pos = position[index];
		}
		return pos;
	
	}
	// Return the nodeFactory
	return nodeFactory;
})
