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
					links:		new Object(),
					//pos:		initPosition(index) ,
					
					//to be complient with force layout:
					x:			initPosition(data.id).x,
					y:			initPosition(data.id).y,
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
	var addLinkFun = function(link) {
		
		// Check this link is not already included:
		// ! Comment this line if we want double link
		//if(! this.doeslinkExist(link.targetNode, link) ){
			// Add the link to the node
			this.links[link.index] = link;
		//}
	}


	// Fetch an abstract per ajax
	var getAbstractFun = function(callback) {
		// If we have an abstract already, call the callback
		if (this.abstract != undefined) callback(this.abstract)

		// If not, then fetch abstract from server
		else {
			$.get("ajax/abstract/" + this.id, {}, function (data) { 
				if (data.success == true) {
					this.abstract = data.abstract;
				} else {
					this.abstract = "Not found";
				}

				if (callback != undefined) callback(this.abstract);
			});
		}
	}


	// Get date from node
	var getDateFun = function() {
		var date		= new Date(parseInt(this.time) + (new Date()).getTimezoneOffset()*60000)
		return date;
	}

	
	// Checks if the link already exists
	var doeslinkExistFun = function(target, link) {
		var test = (target.links[link.index]  != undefined);
		if(test) console.log("no duplicate");
		return test;
	}

	
	// Find initial position of the node, else create it.
	var initPosition = function(id) {
		var pos = {};
		if(position[id] == null){
			
			pos.x = config['graph_width']*Math.random();
			pos.y = config['graph_height']*Math.random();
			
		}else {
			pos = position[id];
		}
		return pos;
	
	}


	// Return the nodeFactory
	return nodeFactory;
})
