
package paper


import breeze.linalg.DenseVector
import breeze.classify
import org.netlib.lapack.LAPACK
import org.netlib.util.intW
import breeze.linalg.support.{CanCopy}




trait BagOfWordsLSI {


 // error is here:
  def compareBoWLSI(paperPos: String, papers : List[Paper], limit : Int) : List[Paper] = {
	  val loadedPapers = if(papers == List()) CacheLoader.load(paperPos, Cache.extended) else papers
	  val matrixOfWeights: breeze.linalg.DenseMatrix[Int] = createTDMatrix(loadedPapers,50)

	  loadedPapers.map(p => {
				// Check that paper isn't already linked
				if (p.meta.get("linked") == None) {
					// Get list of papers that aren't current paper
					val otherPapers = loadedPapers.filter(p != _)
					// Compare to every other paper
					val weights : List[Int] = for (other <- otherPapers) yield getScores(matrixOfWeights,p.index).valueAt(other.index)
					// Make links
					//val links = for ((p,w) <- otherPapers.zip(weights) if w >= limit) yield Link(p.id,w)
					val links = for ((p,w) <- otherPapers.zip(weights) if w >= limit) yield Link(p.index,w)

					// Add links to paper, and set it as linked
					val result = p.setLinks(links).setMeta("linked", "yes")

					// Save result
					Cache.save(result, Cache.linked)

					result
				}
				else p
			})
	}
  
  def getScores(matrixOfScores: breeze.linalg.DenseMatrix[Int], column: Int): DenseVector[Int] ={

	  val matrixOfScoresTranspose = matrixOfScores.t

	  return matrixOfScoresTranspose(::,column)

	}

	//Methods to compute the term-document matrix 
	def tf(term: String, document: Int, counts: Array[Map[java.lang.String,Int]]): Double = {	
			if (counts(document).contains(term)){
				val freq = counts(document)(term)
				val normalizedFreq = freq				
				return normalizedFreq				
			}else{					 
				return 0.0
			}
	}

	//Computing IDF value
	def idf(term: String, datasetSize : Double, counts: Array[Map[java.lang.String,Int]]): Double = {
			// take the logarithm of the quotient of the number of documents by the documents where term t appears
			//convert appearances to a float (to avoid errors)
			var appearances : Double = 0.0
			counts foreach {x => if (x.contains(term)){
									appearances += 1

						   }
			}
			val a = math.log(datasetSize/appearances)  
			return a
	}

	def tfidf(term:String, document: Int, datasetSize : Double, counts: Array[Map[java.lang.String,Int]]) : Double = {
		//Create tfidf matrix
		//tfidf = tf*idf
		val tfidf = tf(term,document,counts)*idf(term,datasetSize,counts)
		return tfidf
	}

	//Creating the matrix:
	def createTDMatrix(papers: List[Paper], approximation: Int): breeze.linalg.DenseMatrix[Int] = {
		val datasetSize = papers.length

		//Initialisation of arrays
		//Array storing the different sources and the different texts
		
		//preprocess the papers:
		val (textsList,counts) = preprocessTexts(papers)
		
		//building dictionary:
		//find unique words in texts
		val dictionary = buildDictionary(textsList, datasetSize)
			
		// we compute the Matrix of scores for the vectors of words for every document
		//construct it as a vector to convert it as a Matrix
		val tfidfVector = new Array[Double](dictionary.length*datasetSize)
		var j = 0 
		for (i <- 0 to dictionary.length*datasetSize-1){
			//compute tfidf value for word i and document j
			//check if we have reached the length of the dictionary we change document and compute values
			if (i % dictionary.length == 0 && i != 0){
				j += 1	
			}
				
			tfidfVector(i) = tfidf(dictionary(i%dictionary.length),j,datasetSize,counts)			
		}

			//transform tfidf vector into a matrix to compute SVD:
			val termDocMatrix = new breeze.linalg.DenseMatrix[Double](dictionary.length,datasetSize,tfidfVector)

			//once having the termDocMatrix, compute the SVD and then compute cosine similarity on the new matrix:
			//SVD method returns simple arrays - need to convert them:
			//Dimensionality reduction:
			val (u,s,v) = svd(termDocMatrix)

			//converting s and v to 2 dimensional arrays:
			var vo = reconstructArray(v,termDocMatrix.cols)
			println(vo.length + " " + vo.transpose.length)
			var so = Array.ofDim[Double](termDocMatrix.rows,termDocMatrix.cols)
			println(so.length + " " + so.transpose.length)
			var count = 0
			var count2 = 0

			// put the vector s into a n*m matrix so form:
			s foreach{e=>	 
			  if(count2 <= termDocMatrix.cols-1){
					so(count)(count2) = e		
					count += 1
					count2 += 1			
			  		}
			}
			// SVD returns the first k values as the k highest values - assuming no need for indices:
			//select values of the indices in the given array s - indices are the 0 to k-1 first values:
			val emptyArray = new Array[Array[Double]](0)
			//keeping k greatest singular values:
			val keptValues = selectElementsOf2dimArray(so,(0 to approximation-1).toList,emptyArray)
			var newKeptValues = new Array[Array[Double]](keptValues.length,keptValues.length)
			var arrayCounter = 0 

			println(keptValues.deep.mkString("\n"))
			val keptVTr = keptValues.transpose
			keptVTr foreach{e =>
			if (arrayCounter <= approximation-1){
				newKeptValues(arrayCounter) = e
				arrayCounter += 1
			}else{
			  Nil
			}
			  }
			println("separation")
			println(newKeptValues.deep.mkString("\n"))

			val emptyArray2 = new Array[Array[Double]](0)
			//Select k=approximation rows from Dt the document matrix where each column represents a document
			var newVo = selectElementsOf2dimArray(vo,(0 to approximation-1).toList,emptyArray2)

			val recomposedMatrix = newVo.transpose.flatten

			//Matrix reduction with an approximation app - if no matrix reduction, use old tfidf method:
			def matchApproximation(app: Int): breeze.linalg.DenseMatrix[Double] = app match{
			  //keep old method if no matrix reduction
			  case termDocMatrix.cols =>  termDocMatrix
			  case 0 => termDocMatrix
			  //else compute with recomposedMatrix	
			  case y => new breeze.linalg.DenseMatrix[Double](app,termDocMatrix.cols,recomposedMatrix)	            
			}

			val newtermDocMatrix = matchApproximation(approximation)

			//compute cosine similarity:		
			val similarityMatrix = computeCosineSimilarity(datasetSize, newtermDocMatrix)
			//normalize weights from 1 to 100
			val maximalWeight = similarityMatrix.max
			val normalizedCosSimilarity = similarityMatrix.map(weight =>{
												((weight*100)/maximalWeight).toInt
															})

			return normalizedCosSimilarity
	}
	
	def buildDictionary(textsList: List[java.lang.String], datasetSize: Int) : List[java.lang.String] ={
		val dictionary = textsList.distinct.sortWith(_<_)
		return dictionary
	}

	def preprocessTexts(papers: List[Paper]): (List[java.lang.String], Array[Map[java.lang.String,Int]]) ={
		val source = new Array[scala.io.BufferedSource](papers.length)
		var text = new Array[java.lang.String](papers.length)
		val occurences = new Array[Map[java.lang.String,Array[java.lang.String]]](papers.length)
		//now we want to have a map between words and the number of occurences
		//create an array for easier manipulation
		val counts = new Array[Map[java.lang.String,Int]](papers.length)		
		//Create an array of lists to store all different lists of keys:
		val countsList = new Array[List[java.lang.String]](papers.length)
		//List holding all the list of strings of all the texts
		var textsList = List[java.lang.String]()
		//reading from every entry of the list:
		for (k <- 0 to papers.length-1){
			text(k) = papers(k).getAbstract.getText		    
			//leave out unecessary characters from the analysis
			text(k) = clean(text(k))
			    
			//Splitting the string into words to add stemming to every single word
			val splitString = text(k).split("\\s")
			var stemmedString = new Array[java.lang.String](splitString.length)
			var i = 0
			splitString	foreach	{e=>
			  						val a = breeze.text.analyze.PorterStemmer.apply(e)
		  							//There is still a blank space at the beginning of string (does not affect output)
		  							stemmedString(i) = a
		  							i+=1
		  							}

			// create a map of the keys of the text with their occurences
			counts(k) = stemmedString.groupBy(x=>x).mapValues(x=>x.length)

			//only working with keys for now, creating a list of keys for every text:
			countsList(k) = counts(k).keys.toList		

			if(k == 0){
			textsList = countsList(k)
			}else{
			textsList = textsList ::: countsList(k)									
			}
		}
		return (textsList, counts)
	}
	
	def computeCosineSimilarity(datasetSize: Int, termDocMatrix: breeze.linalg.DenseMatrix[Double]): breeze.linalg.DenseMatrix[Double]={
	  val similarityMatrix = breeze.linalg.DenseMatrix.zeros[Double](datasetSize,datasetSize)
	  for (i <- 0 to datasetSize-1){
				for (j <- 0 to datasetSize-1){
					if(i==j){
						similarityMatrix(i,j) = -1
					}else{
						//Compute scalar product between two matrices
						val firstColumn = termDocMatrix(0 to termDocMatrix.rows-1,i)
						val secondColumn =  termDocMatrix(0 to termDocMatrix.rows-1,j)
							
						similarityMatrix(i,j) = firstColumn.dot(secondColumn)	 

						//Compute 2nd norm and output cosine similarity
						val firstColumnNorm = firstColumn.norm(2)
						val secondColumnNorm = secondColumn.norm(2)
							
						similarityMatrix(i,j) = similarityMatrix(i,j)/(firstColumnNorm*secondColumnNorm)
						println(i + " and j is " + j )
						println(similarityMatrix(i,j))
					    }
					}
				}
	  return similarityMatrix
	}
	
	def exportMatrixToText(matrix: String) : Unit = {
		val file = new java.io.File("exportToGephi.csv")
		val p = new java.io.PrintWriter(file)
		p.println(matrix)
		p.close
	}
	//remove an element on a given index from a given list:
	def dropIndex[T](xs: List[T], n: Int) = {
		val (l1, l2) = xs splitAt n
		l1 ::: (l2 drop 1)
	}

	def selectElementsOfArray(inputArray: Array[Double], inputIndices: List[Int], returnArray: Array[Double]): Array[Double] = {
		//var returnArray = new Array[Double](inputIndices.length)
		if(inputIndices == Nil){
			inputArray
		}else if(inputIndices.length != 1){		
			//Add element of the input array corresponding to the first index in the index list:
			var newreturnArray = returnArray :+ inputArray(inputIndices(0))		
					selectElementsOfArray(inputArray,dropIndex(inputIndices,0),newreturnArray)
		}else{
			var newreturnArray = returnArray :+ inputArray(inputIndices(0))	
					return newreturnArray					
		}		
	}	

	def dotProduct[T <% Double](as: Iterable[T], bs: Iterable[T]) = {
		require(as.size == bs.size)
		(for ((a, b) <- as zip bs) yield a * b) sum
	}

	def selectElementsOf2dimArray(inputArray: Array[Array[Double]], inputIndices: List[Int], returnArray: Array[Array[Double]]): Array[Array[Double]] = {
		if(inputIndices == Nil){
			inputArray
		}else if(inputIndices.length != 1){		
			//Add element of the input array corresponding to the first index in the index list:
			var newreturnArray = returnArray :+ inputArray(inputIndices(0))		
			selectElementsOf2dimArray(inputArray,dropIndex(inputIndices,0),newreturnArray)
		}else{
			var newreturnArray = returnArray :+ inputArray(inputIndices(0))	
			return newreturnArray					
		}		
	}

	def clean(in : String) = { if (in == null) "" else in.replaceAll("[^A-Za-z_]", " ").toLowerCase
	}
	//computing find method to return the k largest indices of elements in a vector
	//deletes maximal values of original array (to fix). Otherwise works perfectly // 
	def findKMax(inputVector: Array[Double], k: Int, listOfIndex: List[Int]): List[Int]= {
		if(listOfIndex.length < k){
			val maxofArray = inputVector.max
			if (maxofArray != 0){
					var newlistOfIndex = listOfIndex:::List(inputVector.findIndexOf(x => x == maxofArray))				
					//set maximal value to 0 so it does not get taken into account again
					inputVector(inputVector.findIndexOf(x => x == maxofArray)) = 0
					findKMax(inputVector,k, newlistOfIndex)

				}else{
					listOfIndex
				}
		}else{
			return listOfIndex
		}
	}
	// finds the N maximal values (not the indices)
	def topNs(xs: Array[Double], n: Int) = {
		var ss = List[Double]()
		var min = Double.MaxValue
		var len = 0
		xs foreach { e =>
						if (len < n || e > min) {
						ss = (e :: ss).sorted
						min = ss.head
						len += 1
						}
						if (len > n) {
						ss = ss.tail
						min = ss.head
						len -= 1
						}                    
					}
		ss
	} 	

	// to do: fix output
	def indexOftopNs(xs: Array[Double], n: Int) = {
		var ss = List[Int]()
		var min = Double.MaxValue
		var len = 0
		xs foreach { e =>
						if (len < n || e > min) {
							ss = (xs.findIndexOf(x=>x==e) :: ss).sorted
							min = ss.head
							len += 1
						}
						if (len > n) {
							ss = ss.tail
							min = ss.head
							len -= 1
						}                    
		}
		ss
	} 	


	// function converting a DenseVector to a List of Double
	def convertToList(inputVector : DenseVector[Double]) :  List[Double] = {
		val outputVector = List[Double]()
		for(i <- inputVector){
			outputVector:::List(i)
		}
		return outputVector
	}

	//desiredLength = length of the rows (= number of columns)
	def reconstructArray(inputArray: Array[Double], desiredLength: Int)={
		val doubleDimArray = Array.ofDim[Double]((inputArray.length/ desiredLength).toInt,desiredLength)
		var k = 0
		var i = 0
		inputArray foreach { e =>  			 
								if(k < desiredLength){
									doubleDimArray(k)(i) = e
									k = k+1
								}
								if(k == desiredLength){
								k = 0
								i = i+1
								}			  
		}
		doubleDimArray					 
	}
	//redifining svd:

	//modification of the www.netlib.org/lapack/ package, dgesdd method - derived from breeze svd
	def svd(mat: breeze.linalg.DenseMatrix[Double]):(Array[Double],Array[Double],Array[Double]) = {
		//	requireNonEmptyMatrix(mat)

		val m = mat.rows
		val n = mat.cols
		val S = new Array[Double](m min n)
		val U =  new Array[Double](m*m)
		val Vt = new Array[Double](n*n)
		val iwork = new Array[Int](8 * (m min n) )
		val workSize = ( 3
						* scala.math.min(m, n)
						* scala.math.min(m, n)
						+ scala.math.max(scala.math.max(m, n), 4 * scala.math.min(m, n)
						* scala.math.min(m, n) + 4 * scala.math.min(m, n))
						)
		val work = new Array[Double](workSize)
		val info = new intW(0)
		val cm = copy(mat)
		println("im in")
		LAPACK.getInstance.dgesdd(
				"A", m, n,
				cm.data, scala.math.max(1,m),
				S, U , scala.math.max(1,m),
				Vt, scala.math.max(1,n),
				work,work.length,iwork, info)

		if (info.`val` > 0)
			throw new NotConvergedException(NotConvergedException.Iterations)
		else if (info.`val` < 0)
			throw new IllegalArgumentException()

		(U,S,Vt)
	}
	def copy[T](t: T)(implicit canCopy: CanCopy[T]): T = canCopy(t)

	class MatrixEmptyException extends IllegalArgumentException("Matrix is empty")

	class NotConvergedException(val reason: NotConvergedException.Reason, msg: String = "")
	extends RuntimeException(msg)

	object NotConvergedException {
		trait Reason
		object Iterations extends Reason
		object Divergence extends Reason
		object Breakdown extends Reason
	}

	//Scala for java developers: http://blog.scala4java.com/2011/12/matrix-multiplication-in-scala-single.html

	def multiplyArrays(m1: Array[Array[Double]], m2: Array[Array[Double]]) :  Array[Array[Double]] = {
    val res =  Array.ofDim[Double](m1.length, m2(0).length)
    val M1_COLS = m1(0).length
    val M1_ROWS = m1.length
    val M2_COLS = m2(0).length

    @inline def singleThreadedMultiplicationFAST(start_row:Int,  end_row:Int) {
      var col, i  = 0
      var sum = 0.0
      var row = start_row

      // while statements are much faster than for statements
      while(row < end_row){ col = 0
        while(col < M2_COLS){ i = 0; sum = 0
          while(i<M1_COLS){
            sum += m1(row)(i) * m2(i)(col)
            i+=1
          }

          res(row)(col) = sum
          col += 1

        }; row += 1
      }
    }

    (0 until M1_ROWS).par.foreach( i =>
      singleThreadedMultiplicationFAST(i, i+1)
    )

    res

  }

}
