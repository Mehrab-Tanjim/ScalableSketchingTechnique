/**
 * QCRI, sPCA LICENSE
 * sPCA is a scalable implementation of Principal Component Analysis (PCA) on of Spark and MapReduce
 *
 * Copyright (c) 2015, Qatar Foundation for Education, Science and Community Development (on
 * behalf of Qatar Computing Research Institute) having its principle place of business in Doha,
 * Qatar with the registered address P.O box 5825 Doha, Qatar (hereinafter referred to as "QCRI")
 *
*/
package org.qcri.sparkpca;

import java.io.File;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;

import org.apache.hadoop.io.IntWritable;
import org.apache.log4j.Level;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.DoubleDoubleFunction;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.solver.EigenDecomposition;
import org.apache.spark.Accumulator;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.QRDecomposition;
import org.apache.spark.mllib.linalg.SparseVector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;
import org.apache.spark.storage.StorageLevel;
import org.qcri.sparkpca.FileFormat.OutputFormat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import scala.Tuple2;

/**
 * This code provides an implementation of PPCA: Probabilistic Principal
 * Component Analysis based on the paper from Tipping and Bishop:
 * 
 * sPCA: PPCA on top of Spark
 * 
 * 
 * @author Tarek Elgamal
 * 
 */

public class SparkPCA implements Serializable {

	// private final static Logger log =
	// LoggerFactory.getLogger(SparkPCA.class);// getLogger(SparkPCA.class);
	final int MAX_ROUNDS = 100;
	final static double rCond = 1.0E-5d;// what the hell svd factor dunno TODO
										// tune this if needs more precision, it
										// definitely takes more time to get
										// more precision
	private final static boolean CALCULATE_ERR_ATTHEEND = false;
	static double k_plus_one_singular_value = 0;
	static int nClusters = 4;
	static int subsample = 10;
	static String dataset = "Untitled";
	static long startTime, endTime, totalTime;
	public static Stat stat = new Stat();

	public static void main(String[] args) {
		org.apache.log4j.Logger.getLogger("org").setLevel(Level.ERROR);
		org.apache.log4j.Logger.getLogger("akka").setLevel(Level.ERROR);

		// Parsing input arguments
		final String inputPath;
		final String outputPath;
		final int nRows;
		final int nCols;
		final int nPCs;
		final int trials;

		try {
			inputPath = System.getProperty("i");
			if (inputPath == null)
				throw new IllegalArgumentException();
		} catch (Exception e) {
			return;
		}
		try {
			outputPath = System.getProperty("o");
			if (outputPath == null)
				throw new IllegalArgumentException();
		} catch (Exception e) {
			return;
		}

		try {
			nRows = Integer.parseInt(System.getProperty("rows"));
		} catch (Exception e) {
			return;
		}

		try {
			trials = Integer.parseInt(System.getProperty("trials"));
		} catch (Exception e) {
			return;
		}

		try {
			nCols = Integer.parseInt(System.getProperty("cols"));
		} catch (Exception e) {
			return;
		}
		try {

			if (Integer.parseInt(System.getProperty("pcs")) == nCols) {
				nPCs = nCols - 1;
				System.out
						.println("Number of princpal components cannot be equal to number of dimension, reducing by 1");
			} else
				nPCs = Integer.parseInt(System.getProperty("pcs"));
		} catch (Exception e) {
			return;
		}
		/**
		 * Defaults for optional arguments
		 */
		double errRate = 1;
		int maxIterations = 3;
		OutputFormat outputFileFormat = OutputFormat.DENSE;
		int computeProjectedMatrix = 0;

		try {
			errRate = Float.parseFloat(System.getProperty("errSampleRate"));
		} catch (Exception e) {

			int length = String.valueOf(nRows).length();
			if (length <= 4)
				errRate = 1;
			else
				errRate = 1 / Math.pow(10, length - 4);
			// log.warn("error sampling rate set to: errSampleRate=" + errRate);
		}

		try {
			subsample = Integer.parseInt(System.getProperty("subSample"));
			System.out.println("Subsample is set to" + subsample);
		} catch (Exception e) {

		}

		if ((nPCs + subsample) >= nCols) {
			subsample = nCols - nPCs;
			// log.warn("Subsample is set to default");
		}

		try {
			nClusters = Integer.parseInt(System.getProperty("clusters"));
			System.out.println("No of partition is set to" + nClusters);
		} catch (Exception e) {
			// log.warn("Cluster size is set to default: "+nClusters);
		}

		try {
			maxIterations = Integer.parseInt(System.getProperty("maxIter"));
		} catch (Exception e) {
			// log.warn("maximum iterations is set to default: maximum
			// Iterations=" + maxIterations);
		}

		try {
			dataset = System.getProperty("dataset");
		} catch (IllegalArgumentException e) {
			// log.warn("Invalid Format " + System.getProperty("outFmt") + ",
			// Default name for dataset" + dataset + " will be used ");
		} catch (Exception e) {
			// log.warn("Default oname for dataset " + dataset + " will be used
			// ");
		}

		try {
			outputFileFormat = OutputFormat.valueOf(System.getProperty("outFmt"));
		} catch (IllegalArgumentException e) {
			// log.warn("Invalid Format " + System.getProperty("outFmt") + ",
			// Default output format" + outputFileFormat + " will be used ");
		} catch (Exception e) {
			// log.warn("Default output format " + outputFileFormat + " will be
			// used ");
		}

		try {
			computeProjectedMatrix = Integer.parseInt(System.getProperty("computeProjectedMatrix"));
		} catch (Exception e) {
			// log.warn("Projected Matrix will not be computed, the output path
			// will contain the principal components only");
		}

		// Setting Spark configuration parameters
		SparkConf conf = new SparkConf().setAppName("SSVD").setMaster("local[*]");//
																					// TODO
																					// remove
																					// this
																					// part
																					// for
																					// building
		conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
		JavaSparkContext sc = new JavaSparkContext(conf);

		// compute principal components
		computePrincipalComponents(sc, inputPath, outputPath, nRows, nCols, nPCs, trials, maxIterations,
				computeProjectedMatrix);

		// log.info("Principal components computed successfully ");
	}

	public static org.apache.spark.mllib.linalg.Matrix computePrincipalComponents(JavaSparkContext sc, String inputPath,
			String outputPath, final int nRows, final int nCols, final int nPCs, final double errRate,
			final int maxIterations, final int computeProjectedMatrix) {
		
		// Read from sequence file
		JavaPairRDD<IntWritable, VectorWritable> seqVectors = sc.sequenceFile(inputPath, IntWritable.class,
				VectorWritable.class, nClusters);

		// TODO for dense matrix modify here

		// Convert sequence file to RDD<org.apache.spark.mllib.linalg.Vector> of
		// Vectors
		JavaRDD<org.apache.spark.mllib.linalg.Vector> vectors = seqVectors
				.map(new Function<Tuple2<IntWritable, VectorWritable>, org.apache.spark.mllib.linalg.Vector>() {

					public org.apache.spark.mllib.linalg.Vector call(Tuple2<IntWritable, VectorWritable> arg0)
							throws Exception {

						org.apache.mahout.math.Vector mahoutVector = arg0._2.get();
						Iterator<Element> elements = mahoutVector.nonZeroes().iterator();
						ArrayList<Tuple2<Integer, Double>> tupleList = new ArrayList<Tuple2<Integer, Double>>();
						while (elements.hasNext()) {
							Element e = elements.next();
							if (e.index() >= nCols || e.get() == 0)
								continue;
							Tuple2<Integer, Double> tuple = new Tuple2<Integer, Double>(e.index(), e.get());
							tupleList.add(tuple);
						}
						org.apache.spark.mllib.linalg.Vector sparkVector = Vectors.sparse(nCols, tupleList);
						return sparkVector;
					}
				}).persist(StorageLevel.MEMORY_ONLY_SER()); // TODO
															// change
															// later;

		// 1. Mean Job : This job calculates the mean and span of the columns of
		// the input RDD<org.apache.spark.mllib.linalg.Vector>
		final Accumulator<double[]> matrixAccumY = sc.accumulator(new double[nCols], new VectorAccumulatorParam());
		final double[] internalSumY = new double[nCols];
		vectors.foreachPartition(new VoidFunction<Iterator<org.apache.spark.mllib.linalg.Vector>>() {

			public void call(Iterator<org.apache.spark.mllib.linalg.Vector> arg0) throws Exception {
				org.apache.spark.mllib.linalg.Vector yi;
				int[] indices = null;
				int i;
				while (arg0.hasNext()) {
					yi = arg0.next();
					indices = ((SparseVector) yi).indices();
					for (i = 0; i < indices.length; i++) {
						internalSumY[indices[i]] += yi.apply(indices[i]);
					}
				}
				matrixAccumY.add(internalSumY);
			}

		});// End Mean Job

		// Get the sum of column Vector from the accumulator and divide each
		// element by the number of rows to get the mean
		// not best of practice to use non-final variable
		final Vector meanVector = new DenseVector(matrixAccumY.value()).divide(nRows);
		final Broadcast<Vector> br_ym_mahout = sc.broadcast(meanVector);

		

		/************************** SSVD PART *****************************/
		
		// initialize & broadcast a random seed
		org.apache.spark.mllib.linalg.Matrix GaussianRandomMatrix = org.apache.spark.mllib.linalg.Matrices.randn(nCols,
				nPCs + subsample, new java.util.Random(System.currentTimeMillis()));
		final Matrix seedMahoutMatrix=PCAUtils.convertSparkToMahoutMatrix(GaussianRandomMatrix);
		final Broadcast<Matrix> seed = sc.broadcast(seedMahoutMatrix);
		PCAUtils.printMatrixToFile(GaussianRandomMatrix, OutputFormat.DENSE, "Seed");

		final Vector seedMu=seedMahoutMatrix.transpose().times(meanVector);
		final Broadcast<Vector> brSeedMu = sc.broadcast(seedMu);
		
		JavaRDD<org.apache.spark.mllib.linalg.Vector> Y = vectors
				.map(new Function<org.apache.spark.mllib.linalg.Vector, org.apache.spark.mllib.linalg.Vector>() {

					public org.apache.spark.mllib.linalg.Vector call(org.apache.spark.mllib.linalg.Vector arg0)
							throws Exception {
						double[] y=new double[nPCs+subsample];
						int [] indices=((SparseVector)arg0).indices();
						int index;
	    				double value=0;
	    				for(int j=0;j<(nPCs+subsample);j++){
	    					for(int i=0; i< indices.length; i++ ){	 	    				
	 	    					index=indices[i];
	 	    					value+=arg0.apply(index)*seed.value().getQuick(index, j);
	 	    				}
	    					y[j]=value-brSeedMu.value().getQuick(j);
	    					value=0;
	    				}
	 	    			
	     			    return Vectors.dense(y);
						
					}
				});
		
				// QR decomposition of B
		QRDecomposition<RowMatrix, org.apache.spark.mllib.linalg.Matrix> QR = new RowMatrix(Y.rdd()).tallSkinnyQR(true);

		JavaPairRDD<org.apache.spark.mllib.linalg.Vector, org.apache.spark.mllib.linalg.Vector> QnA = QR.Q().rows()
				.toJavaRDD().zip(vectors);
		final Accumulator<double[]> sumQ = sc.accumulator(new double[nPCs+subsample], new VectorAccumulatorParam());
		final Accumulator<double[]> sumQtA = sc.accumulator(new double[(nPCs+subsample)*nCols], new VectorAccumulatorParam());
		

		final double[] sumQPartial = new double[nPCs+subsample];
		final double[] sumQtAPartial = new double[(nPCs+subsample)*nCols];
		
		QnA.foreachPartition(new VoidFunction<Iterator<Tuple2<org.apache.spark.mllib.linalg.Vector,org.apache.spark.mllib.linalg.Vector>>>() {

			@Override
			public void call(Iterator<Tuple2<org.apache.spark.mllib.linalg.Vector, org.apache.spark.mllib.linalg.Vector>> arg0) throws Exception {
				
				Tuple2<org.apache.spark.mllib.linalg.Vector, org.apache.spark.mllib.linalg.Vector> pair;
				org.apache.spark.mllib.linalg.Vector A;
				org.apache.spark.mllib.linalg.Vector Q;
				while(arg0.hasNext()){
					//lol mistake
					pair=arg0.next();
					A = pair._2;
					Q = pair._1;
					
					int row = Q.size();
					int [] indices=((SparseVector)A).indices();
					int index;
					for (int j = 0; j < indices.length;  j++) {
						for (int i = 0; i < row; i++) {
							index=indices[j];
							sumQtAPartial[row *index + i]+=Q.apply(i) * A.apply(index);
						}
					}
					for (int i = 0; i < row; i++) {
						sumQPartial[i]+=Q.apply(i);
					}
					
				}	

				sumQ.add(sumQPartial);
				sumQtA.add(sumQtAPartial);
				
			}
			
		});
		
		org.apache.spark.mllib.linalg.Matrix QtA = org.apache.spark.mllib.linalg.Matrices.dense(nPCs+subsample, nCols, sumQtA.value());

		double[][] QtAArray=new double[nPCs+subsample][nCols];
		
		for(int i=0;i<(nPCs+subsample);i++){
			for(int j=0;j<nCols;j++){
				QtAArray[i][j]=QtA.apply(i,j)-sumQ.value()[i]*br_ym_mahout.value().getQuick(j);
			}
		}
		Matrix B=new DenseMatrix(QtAArray);
		org.apache.mahout.math.SingularValueDecomposition SVD = new org.apache.mahout.math.SingularValueDecomposition(B);
		
		org.apache.spark.mllib.linalg.Matrix V = PCAUtils
				.convertMahoutToSparkMatrix(SVD.getV().viewPart(0, nCols, 0, nPCs));

		PCAUtils.printMatrixToFile(V, OutputFormat.DENSE, outputPath + File.separator + "V");
		
		seed.destroy();
		
		return null;
	}

}