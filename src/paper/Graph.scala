package paper

trait Graphs {

  def getGraph(paperPos:String, papers : List[Paper]) : Graph = {
    println("BEGIN OF GRAPH CREATION")
    val loadedPapers = if(papers == List()) CacheLoader.load(paperPos, Cache.linked) else papers
    // Add all papers as nodes
    val nodes : List[Node] = for (p <- loadedPapers) yield makeNode(p)

    // Then create all edges
    val edges : List[Edge] = for (p <- loadedPapers; e <- makeEdges(p, nodes)) yield e

    println("END OF GRAPH CREATION")
    return new Graph(nodes, edges)
  }

  // MODIFICATION
 def makeNode(paper : Paper) : Node = {
    println("Making node for " + paper.id)
    Node(paper.id, paper.meta("xmlpapertitle"), paper.meta("xmlauthors"), paper.meta("pdf"), paper.meta("xmldate"), paper.meta("xmlroom"))
  }
  
  def makeEdges(paper : Paper, nodes : List[Node]) : List[Edge] = {
    // make edge
    val edges = for (link <- paper.links) yield (makeEdge(paper.index, link))
    // Sort edges by weight and pick the n biggest
    //val l = math.min(, edges.length)
    return edges.filter(_.weight>=30)
  }

  def makeEdge(index : Int, link : Link) : Edge = Edge(index, link.index, link.weight)

}

class Graph(nodes : List[Node], edges : List[Edge]) {

  def save : Unit = {
    val f = new java.io.File("graph.js")
    val p = new java.io.PrintWriter(f)
    p.println(toString)
    p.close
  }

  override def toString : String = {
    var ret : String = ""

    // Add define
    ret += "define([], function () {\n\ndata = {"

    // add nodes
    ret += "\"nodes\":" + nodes.mkString("[\n  ",",\n  ","\n],") + "\n"

    // add edges
    ret += "\"links\":" + edges.mkString("[\n  ",",\n  ","\n]") + "\n\n"

    // End
    ret += "}\n\nreturn data;\n})"

    return ret
  }
}

case class Node(id : Int, title : String, authors : String, pdf : String, date : String, room : String) {
  override def toString : String = {
    var ret : String = "{"
    ret += "\"id\":" + id + ",\n   "
    ret += "\"title\":\"" + Escape(title) + "\",\n   "
    ret += "\"authors\":\"" + Escape(authors) + "\",\n   "
    ret += "\"pdf\":\"" + Escape(pdf) + "\",\n   "
    ret += "\"date\":\"" + Escape(date) + "\",\n   "
    ret += "\"room\":\"" + Escape(room) + "\""
    ret += "}"
    return ret
  }
}

case class Edge(from : Int, to : Int, weight : Double) {
  override def toString : String = "{\"source\":" + from + ",\"target\":" + to + ",\"value\":" + weight + "}"
}

/** Escapes a raw string for use in HTML.*/
object Escape
{
	def apply(s: String) =
	{
		val out = new StringBuilder
		for(i <- 0 until s.length)
		{
			s.charAt(i) match
			{
				case '>' => out.append("&gt;")
				case '&' => out.append("&amp;")
				case '<' => out.append("&lt;")
				case '"' => out.append("&quot;")
				case '\n' => out.append(" ")
				case '\\' => out.append("\\\\")
				case c => out.append(c)
			}
		}
		out.toString
	}
}
