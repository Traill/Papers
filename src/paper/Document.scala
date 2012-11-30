package paper

import net.liftweb.json._
import java.io.File


case class Document(id : String,
                    paper : Paper, 
                    file : File, 
                    links : List[Link], 
                    meta : Map[String, String]) extends AbstractDocument

// TODO: delete meta
case class Paper(title:     Title, 
                 authors:   List[Author], 
                 abstr:     Abstract, 
                 body:      Body, 
                 refs:      List[Reference]) extends AbstractPaper


case class Title(text: String)

case class Author(name: String)

case class Abstract(text: String)

case class Body(text: String)

case class Reference(authors: List[Author], title: Title)

case class Link(id : String, weight : Int)

object Document {

  // Empty document for initialization
  val emptyDoc = Document("", emptyPaper, new File(""), List(), (Map.empty : Map[String,String]))

  // Empty paper for initialization
  val emptyPaper = Paper(Title(""), List(), Abstract(""), Body(""), List())

  // Hint for File
  val hints = new ShortTypeHints(classOf[File] :: Nil) {
    override def serialize: PartialFunction[Any, JObject] = {
      case (f: File) => JObject(JField("file", JString(f.getAbsolutePath)) :: Nil)
    }
  }


  // Implicit values so we can write out a paper
  implicit val formats = DefaultFormats.withHints(hints)

  // Convert document to JSON
  def toJSON(d : Document) : String = Serialization.write(d)


  // Convert JSON to paper
  def fromJSON(json : String) : Document = fromJSON(parse(json))
  def fromJSON(json : JValue) : Document = {

    // Parse document from JSON
    def doc(fs : List[JField], d : Document) : Document = fs match {
      case Nil                                  => d
      case JField("paper", JObject(p)) :: rest  => doc(rest, d.setPaper(paper(p, emptyPaper)))
      case JField("file", f) :: rest            => doc(rest, d.setFile(file(f)))
      case JField("links", JArray(ls)) :: rest  => doc(rest, d.setLinks(links(ls)))
      case JField("meta", JObject(m)) :: rest   => doc(rest, meta(m, d))
      case JField("id", JString(t)) :: rest     => doc(rest, d.setId(t))
      case other                                => throw new Exception("No match in document for field: " + other)
    }


    // Parse file from JSON
    def file(json : JValue) : File = json match {
      case JObject(_::JField("file",JString(path))::Nil) => new File(path)
      case other                                         => throw new Exception("No match in file for field: " + other)
    }



    def paper(fs : List[JField], p : Paper) : Paper = fs match {
      case Nil                                    => p
      case JField("title", JObject(t)) :: rest    => paper(rest, p.setTitle(text(t)))
      case JField("authors", JArray(t)) :: rest   => paper(rest, p.setAuthors(authors(t)))
      case JField("abstr", JObject(t)) :: rest => paper(rest, p.setAbstract(text(t)))
      case JField("body", JObject(t)) :: rest     => paper(rest, p.setBody(text(t)))
      case JField("refs", JArray(t)) :: rest      => paper(rest, p.setReferences(refs(t)))
      case other                                  => throw new Exception("No match in paper for field: " + other)
    }


    // Getting text out of title, abstract and body
    def text(fs : List[JField]) = fs match {
      case JField("text", JString(t)) :: rest        => t
      case other                                  => throw new Exception("No match in textfield (title, abstract, body) for field: " + other)
    }

    // Extracts Authors from JSON
    def authors(as : List[JValue]) : List[Author] = as match {
      case Nil                                          => Nil
      case JObject(JField(_,JString(t)) :: _) :: rest   => Author(t) :: authors(rest)
      case other                                        => throw new Exception("Authors have to be a JString: " + other)
    }

    // Extract References from JSON
    def refs(rs : List[JValue]) = rs.map(r => ref(r))
    def ref(r : JValue) = r match {
      case JObject(List(JField("authors", JArray(as)), JField("title", JObject(t))))    => Reference(authors(as), Title(text(t)))
      case otherwise                                                                    => throw new Exception("malformed reference: " + otherwise)
    }

    // Extract Links from JSON
    def links(ls : List[JValue]) = ls.map(l => link(l))
    def link(l : JValue) = l match {
      case JObject(List(JField("id", JString(id)), JField("weight", JInt(weight)))) => Link(id, weight.toInt)
      case otherwise                                                           => throw new Exception("malformed link: " + otherwise)
    }

    // Extract Meta from JSON
    def meta(ms : List[JField], d : Document) : Document = ms match {
      case Nil                                  => d
      case JField(key, JString(value)) :: rest  => meta(rest, d.setMeta(key -> value))
      case otherwise                            => throw new Exception("Wrong meta format: " + otherwise)
    }

    // Check if we have a JObject
    json match {
      case JObject(fs)  => doc(fs, emptyDoc)
      case otherwise    => throw new Exception("Can't parse JSON " + otherwise)
    }
  }


  // Try to clean the data a little
  //def clean(p : Paper) = Paper(p.title, p.authors.filter(a => a.name.length > 4), p.abstr, p.body, cleanRefs(p.refs), p.meta, p.links)
  def cleanRefs(refs : List[Reference]) = refs.map(r => Reference(r.authors.filter(a => a.name.stripMargin.length > 0), r.title))


  // Get a list of all the distinct names in the paper authors and references
  def getDistinctNames(p : Paper) : List[String] = {
    val as = p.authors ::: p.refs.flatMap(r => r.authors)
    as.map(a => a.name).distinct
  }
}

abstract class AbstractDocument {
  val id : String
  val paper : Paper
  val file : File
  val links : List[Link]
  val meta : Map[String, String]

  def setId(newId : String) : Document = Document(newId, paper, file, links, meta)
  def setPaper(p : Paper) : Document = Document(id, p, file, links, meta)
  def setFile(f : File) : Document = Document(id, paper, f, links, meta)
  def setLinks(ls : List[Link]) : Document = Document(id, paper, file, ls, meta)
  def setMeta(m : (String, String)) : Document = Document(id, paper, file, links, meta + m)
  def setMeta(m : Map[String, String]) : Document = Document(id, paper, file, links, meta ++ m)
  def hasMeta(l : String) : Boolean = meta.contains(l)
}


// Abstract class for implementing a few helper methods
abstract class AbstractPaper {
  val title : Title;
  val authors : List[Author];
  val abstr : Abstract;
  val body : Body;
  val refs : List[Reference];

  override def toString : String = "Paper: " + title

  def setTitle(t : String) : Paper =
    Paper(Title(t), authors, abstr, body, refs)
  def setTitle(t : Title) : Paper =
    Paper(t, authors, abstr, body, refs)

  def setAuthors(as : List[Author]) : Paper =
    Paper(title, as, abstr, body, refs)

  def setAbstract(newAbstract : String) : Paper = 
    Paper(title, authors, Abstract(newAbstract), body, refs)
  def setAbstract(newAbstract : Abstract) : Paper = 
    Paper(title, authors, newAbstract, body, refs)
    
  def setBody(newBody : String) : Paper = 
    Paper(title, authors, abstr, Body(newBody), refs)
  def setBody(newBody : Body) : Paper = 
    Paper(title, authors, abstr, newBody, refs)
    
  def setReferences(newRefs : List[Reference]) : Paper = 
    Paper(title, authors, abstr, body, newRefs)
}
