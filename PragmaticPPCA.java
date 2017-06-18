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
import java.io.FileNotFoundException;
import java.io.Serializable;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

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
 * @author Mehrab Tanjim
 * 
 */

public class PragmaticPPCA implements Serializable {

	private final static Logger log = LoggerFactory.getLogger(PragmaticPPCA.class);// getLogger(SparkPCA.class);


	static String dataset = "Untitled";
	static long startTime, endTime, totalTime;
	public static int nClusters = 4;
	public static Stat stat = new Stat();

	public static void main(String[] args) throws FileNotFoundException {
		org.apache.log4j.Logger.getLogger("org").setLevel(Level.ERROR);
		org.apache.log4j.Logger.getLogger("akka").setLevel(Level.ERROR);

		// Parsing input arguments
		final String inputPath;
		final String outputPath;
		final int nRows;
		final int nCols;
		final int nPCs;
		final int q;// default
		final double k_plus_one_singular_value;
		final double tolerance;
		final int subsample;
		final int subsampleNorm;

		try {
			inputPath = System.getProperty("i");
			if (inputPath == null)
				throw new IllegalArgumentException();
		} catch (Exception e) {
			printLogMessage("i");
			return;
		}
		try {
			outputPath = System.getProperty("o");
			if (outputPath == null)
				throw new IllegalArgumentException();
		} catch (Exception e) {
			printLogMessage("o");
			return;
		}

		try {
			nRows = Integer.parseInt(System.getProperty("rows"));
		} catch (Exception e) {
			printLogMessage("rows");
			return;
		}

		try {
			nCols = Integer.parseInt(System.getProperty("cols"));
		} catch (Exception e) {
			printLogMessage("cols");
			return;
		}

		try {
			k_plus_one_singular_value = Double.parseDouble(System.getProperty("kSingularValue"));
		} catch (Exception e) {
			printLogMessage("kSingularValue");
			return;
		}

		try {
			tolerance = Double.parseDouble(System.getProperty("tolerance"));
		} catch (Exception e) {
			printLogMessage("tolerance");
			return;
		}

		try {
			subsample = Integer.parseInt(System.getProperty("subSample"));
			System.out.println("Subsample is set to" + subsample);
		} catch (Exception e) {
			printLogMessage("subsample");
			return;
		}
		
		try {
			subsampleNorm = Integer.parseInt(System.getProperty("subSampleNorm"));
			System.out.println("SubsampleNorm is set to" + subsampleNorm);
		} catch (Exception e) {
			printLogMessage("subsampleNorm");
			return;
		}

		try {
			q = Integer.parseInt(System.getProperty("q"));
			System.out.println("No of q is set to" + q);
		} catch (Exception e) {
			printLogMessage("q");
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
			printLogMessage("pcs");
			return;
		}
		/**
		 * Defaults for optional arguments
		 */
		int maxIterations = 50;
		OutputFormat outputFileFormat = OutputFormat.DENSE;
		int calculateError = 0;

		try {
			nClusters = Integer.parseInt(System.getProperty("clusters"));
			System.out.println("No of partition is set to" + nClusters);
		} catch (Exception e) {
			log.warn("Cluster size is set to default: " + nClusters);
		}

		try {
			maxIterations = Integer.parseInt(System.getProperty("maxIter"));
		} catch (Exception e) {
			log.warn("maximum iterations is set to default: maximum	Iterations=" + maxIterations);
		}

		try {
			dataset = System.getProperty("dataset");
		} catch (IllegalArgumentException e) {
			log.warn("Invalid Format " + System.getProperty("outFmt") + ", Default name for dataset" + dataset
					+ " will be used ");
		} catch (Exception e) {
			log.warn("Default oname for dataset " + dataset + " will be used ");
		}

		try {
			outputFileFormat = OutputFormat.valueOf(System.getProperty("outFmt"));
		} catch (IllegalArgumentException e) {
			log.warn("Invalid Format " + System.getProperty("outFmt") + ", Default output format" + outputFileFormat
					+ " will be used ");
		} catch (Exception e) {
			log.warn("Default output format " + outputFileFormat + " will be used ");
		}

		try {
			calculateError = Integer.parseInt(System.getProperty("calculateError"));
		} catch (Exception e) {
			log.warn(
					"Projected Matrix will not be computed, the output path will contain the principal components only");
		}

		// Setting Spark configuration parameters
		SparkConf conf = new SparkConf().setAppName("pragmaticPPCA");//.setMaster("local[*]");//
		// TODO
		// remove
		// this
		// part
		// for
		// building
		conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
		conf.set("spark.kryoserializer.buffer.max", "128m");
		JavaSparkContext sc = new JavaSparkContext(conf);

		// compute principal components
		computePrincipalComponents(sc, inputPath, outputPath, nRows, nCols, nPCs, subsample, tolerance,
				k_plus_one_singular_value, q, maxIterations, calculateError, subsampleNorm);

		// log.info("Principal components computed successfully ");
	}

	/**
	 * Compute principal component analysis where the input is a path for a
	 * hadoop sequence File <IntWritable key, VectorWritable value>
	 * 
	 * @param sc
	 *            Spark context that contains the configuration parameters and
	 *            represents connection to the cluster (used to create RDDs,
	 *            accumulators and broadcast variables on that cluster)
	 * @param inputPath
	 *            Path to the sequence file that represents the input matrix
	 * @param nRows
	 *            Number of rows in input Matrix
	 * @param nCols
	 *            Number of columns in input Matrix
	 * @param nPCs
	 *            Number of desired principal components
	 * @param errRate
	 *            The sampling rate that is used for computing the
	 *            reconstruction error
	 * @param maxIterations
	 *            Maximum number of iterations before terminating
	 * @return Matrix of size nCols X nPCs having the desired principal
	 *         components
	 * @throws FileNotFoundException
	 */
	public static org.apache.spark.mllib.linalg.Matrix computePrincipalComponents(JavaSparkContext sc, String inputPath,
			String outputPath, final int nRows, final int nCols, final int nPCs, final int subsample,
			final double tolerance, final double k_plus_one_singular_value, final int q, final int maxIterations,
			final int calculateError, final int subsampleNorm) throws FileNotFoundException {

		/**
		 * preprocess the data
		 * 
		 * @param nClusters
		 * 
		 */
		startTime = System.currentTimeMillis();

		// Read from sequence file
		JavaPairRDD<IntWritable, VectorWritable> seqVectors = sc.sequenceFile(inputPath, IntWritable.class,
				VectorWritable.class, nClusters);

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

		// 2. Frobenious Norm Job : Obtain Frobenius norm of the input
		// RDD<org.apache.spark.mllib.linalg.Vector>
		/**
		 * To compute the norm2 of a sparse matrix, iterate over sparse items
		 * and sum square of the difference. After processing each row, add the
		 * sum of the meanSquare of the zero-elements that were ignored in the
		 * sparse iteration.
		 */
		double meanSquareSumTmp = 0;
		for (int i = 0; i < br_ym_mahout.value().size(); i++) {
			double element = br_ym_mahout.value().getQuick(i);
			meanSquareSumTmp += element * element;
		}
		final double meanSquareSum = meanSquareSumTmp;
		final Accumulator<Double> doubleAccumNorm2 = sc.accumulator(0.0);
		vectors.foreach(new VoidFunction<org.apache.spark.mllib.linalg.Vector>() {

			public void call(org.apache.spark.mllib.linalg.Vector yi) throws Exception {
				double norm2 = 0;
				double meanSquareSumOfZeroElements = meanSquareSum;
				int[] indices = ((SparseVector) yi).indices();
				int i;
				int index;
				double v;
				// iterate over non
				for (i = 0; i < indices.length; i++) {
					index = indices[i];
					v = yi.apply(index);
					double mean = br_ym_mahout.value().getQuick(index);
					double diff = v - mean;
					diff *= diff;

					// cancel the effect of the non-zero element in
					// meanSquareSum
					meanSquareSumOfZeroElements -= mean * mean;
					norm2 += diff;
				}
				// For all all zero items, the following has the sum of mean
				// square
				norm2 += meanSquareSumOfZeroElements;
				doubleAccumNorm2.add(norm2);
			}

		});// end Frobenious Norm Job

		double norm2 = doubleAccumNorm2.value();

		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;

		stat.preprocessTime = (double) totalTime / 1000.0;

		stat.totalRunTime = stat.preprocessTime;

		stat.appName = "pragmaticPPCA";
		stat.dataSet = dataset;
		stat.nRows = nRows;
		stat.nCols = nCols;

		// compute principal components
		computePrincipalComponents(sc, vectors, br_ym_mahout, meanVector, norm2, outputPath, nRows, nCols, nPCs,
				subsample, tolerance, k_plus_one_singular_value, q, maxIterations,
				calculateError,subsampleNorm);

		// count the average ppca runtime

		for (int j = 0; j < stat.ppcaIterTime.size(); j++) {
			stat.avgppcaIterTime += stat.ppcaIterTime.get(j);
		}
		stat.avgppcaIterTime /= stat.ppcaIterTime.size();

		// save statistics
		PCAUtils.printStatToFile(stat, outputPath);

		return null;
	}

	/**
	 * Compute principal component analysis where the input is an
	 * RDD<org.apache.spark.mllib.linalg.Vector> of vectors such that each
	 * vector represents a row in the matrix
	 * 
	 * @param sc
	 *            Spark context that contains the configuration parameters and
	 *            represents connection to the cluster (used to create RDDs,
	 *            accumulators and broadcast variables on that cluster)
	 * @param vectors
	 *            An RDD of vectors representing the rows of the input matrix
	 * @param nRows
	 *            Number of rows in input Matrix
	 * @param nCols
	 *            Number of columns in input Matrix
	 * @param nPCs
	 *            Number of desired principal components
	 * @param errRate
	 *            The sampling rate that is used for computing the
	 *            reconstruction error
	 * @param maxIterations
	 *            Maximum number of iterations before terminating
	 * @return Matrix of size nCols X nPCs having the desired principal
	 *         components
	 * @throws FileNotFoundException
	 */
	public static org.apache.spark.mllib.linalg.Matrix computePrincipalComponents(JavaSparkContext sc,
			JavaRDD<org.apache.spark.mllib.linalg.Vector> vectors, final Broadcast<Vector> br_ym_mahout,
			final Vector meanVector, double norm2, String outputPath, final int nRows, final int nCols, final int nPCs,
			final int subsample, final double tolerance, final double k_plus_one_singular_value, final int q,
			final int maxIterations, final int calculateError, final int subsampleNorm) throws FileNotFoundException {

		startTime = System.currentTimeMillis();

		/************************** SSVD PART *****************************/

		/**
		 * Sketch dimension ,S=s Sketched matrix, B=A*S; QR
		 * decomposition, Q=qr(B); SV decomposition, [~,s,V]=svd(Q);
		 */

		// initialize & broadcast a random seed
		// org.apache.spark.mllib.linalg.Matrix GaussianRandomMatrix =
		// org.apache.spark.mllib.linalg.Matrices.randn(nCols,
		// nPCs + subsample, new SecureRandom());
		// //PCAUtils.printMatrixToFile(GaussianRandomMatrix,
		// OutputFormat.DENSE, outputPath+File.separator+"Seed");
		// final Matrix seedMahoutMatrix =
		// PCAUtils.convertSparkToMahoutMatrix(GaussianRandomMatrix);
		/**
		 * Sketch dimension ,S=s Sketched matrix, B=A*S; QR
		 * decomposition, Q=qr(B); SV decomposition, [~,s,V]=svd(Q);
		 */

		// initialize & broadcast a random seed
		 org.apache.spark.mllib.linalg.Matrix GaussianRandomMatrix =
		 org.apache.spark.mllib.linalg.Matrices.randn(nCols,
		 nPCs + subsample, new SecureRandom());
//		  PCAUtils.printMatrixToFile(GaussianRandomMatrix,
//		  OutputFormat.DENSE,
//		  outputPath+File.separator+"Seed");
		 Matrix B = PCAUtils.convertSparkToMahoutMatrix(GaussianRandomMatrix);
//		Matrix GaussianRandomMatrix = PCAUtils.randomValidationMatrix(nCols, nPCs + subsample);
//		Matrix B = GaussianRandomMatrix;
		// PCAUtils.printMatrixToFile(PCAUtils.convertMahoutToSparkMatrix(GaussianRandomMatrix),
		// OutputFormat.DENSE, outputPath+File.separator+"Seed");

		Matrix V = null;
		org.apache.mahout.math.SingularValueDecomposition SVD = null;

		final Broadcast<Matrix> seed = sc.broadcast(B);

		Vector seedMu = B.transpose().times(meanVector);
		final Broadcast<Vector> brSeedMu = sc.broadcast(seedMu);
				
	   	
		final int s=nPCs+subsample;
		final Accumulator<double[][]> matrixAccumYtY = sc.accumulator(new double[s][s],
				new MatrixAccumulatorParam());
		final Accumulator<double[]> matrixAccumY = sc.accumulator(new double[s], new VectorAccumulatorParam());

		/*
		 * Initialize the output matrices and vectors once in order to avoid
		 * generating massive intermediate data in the workers
		 */
		final double[][] resArrayYtY = new double[s][s];
		final double[] resArrayY = new double[s];

		/*
		 * Used to sum the vectors in one partition.
		 */
		final double[][] internalSumYtY = new double[s][s];
		final double[] internalSumY = new double[s];

		vectors.foreachPartition(new VoidFunction<Iterator<org.apache.spark.mllib.linalg.Vector>>() {

			public void call(Iterator<org.apache.spark.mllib.linalg.Vector> arg0) throws Exception {
				org.apache.spark.mllib.linalg.Vector yi;
				while (arg0.hasNext()) {
					yi = arg0.next();

					/*
					 * Perform in-memory matrix multiplication xi = yi' *
					 * Y2X
					 */
					PCAUtils.sparseVectorTimesMatrix(yi, seed.value(), resArrayY);

					// get only the sparse indices
				
					PCAUtils.outerProductArrayInput(resArrayY, brSeedMu.value(), resArrayY,
							brSeedMu.value(), resArrayYtY);
					int i, j;

					// add the sparse indices only
					
					for (i = 0; i < s; i++) {
						internalSumY[i] += resArrayY[i];
						for (j = 0; j < s; j++) {
							internalSumYtY[i][j] += resArrayYtY[i][j];
							resArrayYtY[i][j] = 0; // reset it
						}

					}
				}
				matrixAccumY.add(internalSumY);
				matrixAccumYtY.add(internalSumYtY);
			}

		});// end X'X and Y'X Job

		/*
		 * Get the values of the accumulators.
		 */
		Matrix centralYtY = new DenseMatrix(matrixAccumYtY.value());
		Vector centralSumY = new DenseVector(matrixAccumY.value());

		/*
		 * Mi = (Yi-Ym)' x (Xi-Xm) = Yi' x (Xi-Xm) - Ym' x (Xi-Xm)
		 * 
		 * M = Sum(Mi) = Sum(Yi' x (Xi-Xm)) - Ym' x (Sum(Xi)-N*Xm)
		 * 
		 * The first part is done in the previous job and the second in the
		 * following method
		 */
		centralYtY = PCAUtils.updateXtXAndYtx(centralYtY, centralSumY, seedMu, seedMu, nRows);
		
		seed.destroy();
		
		brSeedMu.destroy();
		
		
		
		
		Matrix R=new org.apache.mahout.math.CholeskyDecomposition(centralYtY,false).getL().transpose();			
			
		R = PCAUtils.inv(R);
		
		Matrix Seed = B.times(R);

		seedMu = Seed.transpose().times(meanVector);
		final Broadcast<Vector> brSeedMu_R = sc.broadcast(seedMu);
		// System.out.println(brSeedMu.value().getQuick(5));

		final Broadcast<Matrix> seed_R = sc.broadcast(Seed);		
		
		
		
		final Accumulator<double[][]> matrixAccumAtQ = sc.accumulator(new double[nCols][s],
				new MatrixAccumulatorParam());
		final Accumulator<double[]> matrixAccumQ = sc.accumulator(new double[s], new VectorAccumulatorParam());

		/*
		 * Initialize the output matrices and vectors once in order to avoid
		 * generating massive intermediate data in the workers
		 */
		final double[][] resArrayAtQ = new double[nCols][s];
		final double[] resArrayQ = new double[s];

		/*
		 * Used to sum the vectors in one partition.
		 */
		final double[][] internalSumAtQ = new double[nCols][s];
		final double[] internalSumQ = new double[s];

		vectors.foreachPartition(new VoidFunction<Iterator<org.apache.spark.mllib.linalg.Vector>>() {

			public void call(Iterator<org.apache.spark.mllib.linalg.Vector> arg0) throws Exception {
				org.apache.spark.mllib.linalg.Vector yi;
				while (arg0.hasNext()) {
					yi = arg0.next();

					/*
					 * Perform in-memory matrix multiplication xi = yi' *
					 * Y2X
					 */
					PCAUtils.sparseVectorTimesMatrix(yi, seed_R.value(), resArrayQ);

					// get only the sparse indices
					int[] indices = ((SparseVector) yi).indices();

					PCAUtils.outerProductWithIndices(yi, br_ym_mahout.value(), resArrayQ, brSeedMu_R.value(),
							resArrayAtQ, indices);
					
					int i, j, rowIndexYtX;

					// add the sparse indices only
					for (i = 0; i < indices.length; i++) {
						rowIndexYtX = indices[i];
						for (j = 0; j < s; j++) {
							internalSumAtQ[rowIndexYtX][j] += resArrayAtQ[rowIndexYtX][j];
							resArrayAtQ[rowIndexYtX][j] = 0; // reset it
						}

					}
					for (i = 0; i < s; i++) {
						internalSumQ[i] += resArrayQ[i];
						

					}
				}
				matrixAccumQ.add(internalSumQ);
				matrixAccumAtQ.add(internalSumAtQ);
			}

		});// end X'X and Y'X Job
		seed_R.destroy();
		
		brSeedMu_R.destroy();
		/*
		 * Get the values of the accumulators.
		 */
		Matrix centralAtQ = new DenseMatrix(matrixAccumAtQ.value());
		Vector centralSumQ = new DenseVector(matrixAccumQ.value());
		
		centralAtQ = PCAUtils.updateXtXAndYtx(centralAtQ, centralSumQ, br_ym_mahout.value(), seedMu, nRows);
		
		B = centralAtQ.transpose();
		
		SVD = new org.apache.mahout.math.SingularValueDecomposition(B);

		V = SVD.getV().viewPart(0, nCols, 0, nPCs);

		// clean up

		/****************************
		 * END OF SSVD
		 ***********************************/

		/**
		 * construct smart guess
		 */

		// calculate ss first

		double ss = 0;
		// from k+1 to s
		double[] latent = new double[nPCs + subsample];

		for (int i = 0; i < (nPCs + subsample); i++) {
			latent[i] = Math.pow(SVD.getS().get(i, i), 2) / (nRows - 1);
		}

		for (int i = nPCs; i < (nPCs + subsample); i++) {
			ss += latent[i];
		}
		
		if(subsample==0)
			ss=PCAUtils.randSS();
		else 
			ss /= subsample;

		// calculate W
		// (L_tilde-ss_tilde*eye(k)).^0.5
		double[] L_ss = new double[nPCs];
		for (int i = 0; i < nPCs; i++) {
			// column major matrix, doesnt matter though symmertrical
			L_ss[i] = Math.pow((latent[i] - ss), 0.5);
		}

		double[][] centralCarray = new double[nCols][nPCs];
		for (int i = 0; i < nCols; i++) {
			for (int j = 0; j < nPCs; j++) {
				centralCarray[i][j] = V.getQuick(i, j) * L_ss[j];
			}
		}

		Matrix centralC = new DenseMatrix(centralCarray);

		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;
		stat.sketchTime = (double) totalTime / 1000.0;
		stat.totalRunTime += stat.sketchTime;

		System.out.println("Rows: " + nRows + ", Columns " + nCols);

		startTime = System.currentTimeMillis();

		double spectral_error;
		double error;

		// initial CtC
		Matrix centralCtC = centralC.transpose().times(centralC);

		Matrix centralY2X = null;
		// -------------------------- EM Iterations
		// while count
		int round = 0;
		double prevObjective = Double.MAX_VALUE;
		// double error = 0;
		double relChangeInObjective = Double.MAX_VALUE;
		double prevError = Double.MAX_VALUE;
		final float threshold = 0.00001f;// not changing much
		double target_error = tolerance; // 95% accuracy

		for (; (round < maxIterations && relChangeInObjective > threshold && prevError > target_error); round++) {

			// Sx = inv( ss * eye(d) + CtC );
			Matrix centralSx = centralCtC.clone();

			// Sx = inv( eye(d) + CtC/ss );
			centralSx.viewDiagonal().assign(Functions.plus(ss));
			centralSx = PCAUtils.inv(centralSx);

			// X = Y * C * Sx' => Y2X = C * Sx'
			centralY2X = centralC.times(centralSx.transpose());

			// Broadcast Y2X because it will be used in several jobs and several
			// iterations.
			final Broadcast<Matrix> br_centralY2X = sc.broadcast(centralY2X);

			// Xm = Ym * Y2X
			Vector xm_mahout = new DenseVector(nPCs);
			xm_mahout = PCAUtils.denseVectorTimesMatrix(br_ym_mahout.value(), centralY2X, xm_mahout);

			// Broadcast Xm because it will be used in several iterations.
			final Broadcast<Vector> br_xm_mahout = sc.broadcast(xm_mahout);
			// We skip computing X as we generate it on demand using Y and Y2X

			// 3. X'X and Y'X Job: The job computes the two matrices X'X and Y'X
			/**
			 * Xc = Yc * MEM (MEM is the in-memory broadcasted matrix Y2X)
			 * 
			 * XtX = Xc' * Xc
			 * 
			 * YtX = Yc' * Xc
			 * 
			 * It also considers that Y is sparse and receives the mean vectors
			 * Ym and Xm separately.
			 * 
			 * Yc = Y - Ym
			 * 
			 * Xc = X - Xm
			 * 
			 * Xc = (Y - Ym) * MEM = Y * MEM - Ym * MEM = X - Xm
			 * 
			 * XtX = (X - Xm)' * (X - Xm)
			 * 
			 * YtX = (Y - Ym)' * (X - Xm)
			 * 
			 */
			final Accumulator<double[][]> matrixAccumXtx = sc.accumulator(new double[nPCs][nPCs],
					new MatrixAccumulatorParam());
			final Accumulator<double[][]> matrixAccumYtx = sc.accumulator(new double[nCols][nPCs],
					new MatrixAccumulatorParam());
			final Accumulator<double[]> matrixAccumX = sc.accumulator(new double[nPCs], new VectorAccumulatorParam());

			/*
			 * Initialize the output matrices and vectors once in order to avoid
			 * generating massive intermediate data in the workers
			 */
			final double[][] resArrayYtX = new double[nCols][nPCs];
			final double[][] resArrayXtX = new double[nPCs][nPCs];
			final double[] resArrayX = new double[nPCs];

			/*
			 * Used to sum the vectors in one partition.
			 */
			final double[][] internalSumYtX = new double[nCols][nPCs];
			final double[][] internalSumXtX = new double[nPCs][nPCs];
			final double[] internalSumX = new double[nPCs];

			vectors.foreachPartition(new VoidFunction<Iterator<org.apache.spark.mllib.linalg.Vector>>() {

				public void call(Iterator<org.apache.spark.mllib.linalg.Vector> arg0) throws Exception {
					org.apache.spark.mllib.linalg.Vector yi;
					while (arg0.hasNext()) {
						yi = arg0.next();

						/*
						 * Perform in-memory matrix multiplication xi = yi' *
						 * Y2X
						 */
						PCAUtils.sparseVectorTimesMatrix(yi, br_centralY2X.value(), resArrayX);

						// get only the sparse indices
						int[] indices = ((SparseVector) yi).indices();

						PCAUtils.outerProductWithIndices(yi, br_ym_mahout.value(), resArrayX, br_xm_mahout.value(),
								resArrayYtX, indices);
						PCAUtils.outerProductArrayInput(resArrayX, br_xm_mahout.value(), resArrayX,
								br_xm_mahout.value(), resArrayXtX);
						int i, j, rowIndexYtX;

						// add the sparse indices only
						for (i = 0; i < indices.length; i++) {
							rowIndexYtX = indices[i];
							for (j = 0; j < nPCs; j++) {
								internalSumYtX[rowIndexYtX][j] += resArrayYtX[rowIndexYtX][j];
								resArrayYtX[rowIndexYtX][j] = 0; // reset it
							}

						}
						for (i = 0; i < nPCs; i++) {
							internalSumX[i] += resArrayX[i];
							for (j = 0; j < nPCs; j++) {
								internalSumXtX[i][j] += resArrayXtX[i][j];
								resArrayXtX[i][j] = 0; // reset it
							}

						}
					}
					matrixAccumX.add(internalSumX);
					matrixAccumXtx.add(internalSumXtX);
					matrixAccumYtx.add(internalSumYtX);
				}

			});// end X'X and Y'X Job

			/*
			 * Get the values of the accumulators.
			 */
			Matrix centralYtX = new DenseMatrix(matrixAccumYtx.value());
			Matrix centralXtX = new DenseMatrix(matrixAccumXtx.value());
			Vector centralSumX = new DenseVector(matrixAccumX.value());

			/*
			 * Mi = (Yi-Ym)' x (Xi-Xm) = Yi' x (Xi-Xm) - Ym' x (Xi-Xm)
			 * 
			 * M = Sum(Mi) = Sum(Yi' x (Xi-Xm)) - Ym' x (Sum(Xi)-N*Xm)
			 * 
			 * The first part is done in the previous job and the second in the
			 * following method
			 */
			centralYtX = PCAUtils.updateXtXAndYtx(centralYtX, centralSumX, br_ym_mahout.value(), xm_mahout, nRows);
			centralXtX = PCAUtils.updateXtXAndYtx(centralXtX, centralSumX, xm_mahout, xm_mahout, nRows);

			// XtX = X'*X + ss * Sx
			final double finalss = ss;
			centralXtX.assign(centralSx, new DoubleDoubleFunction() {
				@Override
				public double apply(double arg1, double arg2) {
					return arg1 + finalss * nRows * arg2;
				}
			});

			// C = (Ye'*X) / SumXtX;
			Matrix invXtX_central = PCAUtils.inv(centralXtX);
			centralC = centralYtX.times(invXtX_central);
			centralCtC = centralC.transpose().times(centralC);

			// Compute new value for ss
			// ss = ( sum(sum(Ye.^2)) + trace(XtX*CtC) - 2sum(XiCtYit))/(N*D);

			// 4. Variance Job: Computes part of variance that requires a
			// distributed job
			/**
			 * xcty = Sum (xi * C' * yi')
			 * 
			 * We also regenerate xi on demand by the following formula:
			 * 
			 * xi = yi * y2x
			 * 
			 * To make it efficient for sparse matrices that are not
			 * mean-centered, we receive the mean separately:
			 * 
			 * xi = (yi - ym) * y2x = yi * y2x - xm, where xm = ym*y2x
			 * 
			 * xi * C' * (yi-ym)' = xi * ((yi-ym)*C)' = xi * (yi*C - ym*C)'
			 * 
			 */

			double ss2 = PCAUtils.trace(centralXtX.times(centralCtC));
			final double[] resArrayYmC = new double[centralC.numCols()];
			PCAUtils.denseVectorTimesMatrix(meanVector, centralC, resArrayYmC);
			final Broadcast<Matrix> br_centralC = sc.broadcast(centralC);
			final double[] resArrayYiC = new double[centralC.numCols()];
			final Accumulator<Double> doubleAccumXctyt = sc.accumulator(0.0);
			vectors.foreach(new VoidFunction<org.apache.spark.mllib.linalg.Vector>() {
				public void call(org.apache.spark.mllib.linalg.Vector yi) throws Exception {
					PCAUtils.sparseVectorTimesMatrix(yi, br_centralY2X.value(), resArrayX);
					PCAUtils.sparseVectorTimesMatrix(yi, br_centralC.value(), resArrayYiC);

					PCAUtils.subtract(resArrayYiC, resArrayYmC);
					double dotRes = PCAUtils.dot(resArrayX, resArrayYiC);
					doubleAccumXctyt.add(dotRes);
				}
			}); // end Variance Job

			double xctyt = doubleAccumXctyt.value();
			ss = (norm2 + ss2 - 2 * xctyt) / (nRows * nCols);
			// log.info("SSSSSSSSSSSSSSSSSSSSSSSSSSSS " + ss + " (" + norm2 + "
			// + "+ ss2 + " -2* " + xctyt);

			
			// log.info("Objective: %.6f relative change: %.6f \n",
			// objective,relChangeInObjective);

			endTime = System.currentTimeMillis();
			totalTime = endTime - startTime;
			stat.ppcaIterTime.add((double) totalTime / 1000.0);
			stat.totalRunTime += (double) totalTime / 1000.0;
			
			double objective = ss;
			relChangeInObjective = Math.abs(1 - objective / prevObjective);
			prevObjective = objective;

			if (calculateError==1) {
				// log.info("Computing the error at round " + round + " ...");
				System.out.println("Computing the error at round " + round + " ...");

				// the following subsample is fixed
				spectral_error = norm(sc, vectors, nRows, nCols, 1, subsampleNorm, q, meanVector, centralC);
				error = (spectral_error - k_plus_one_singular_value) / k_plus_one_singular_value;

				stat.errorList.add((Double) error);
				// log.info("... end of computing the error at round " + round +
				// " And error=" + error);
				System.out.println("... end of computing the error at round " + round + " error=" + error);
				prevError = error;
			}

			/**
			 * reinitialize
			 */
			startTime = System.currentTimeMillis();
		}
		// return the actual PC not the principal subspace
		stat.nIter = round;
		return PCAUtils
				.convertMahoutToSparkMatrix(new org.apache.mahout.math.SingularValueDecomposition(centralC).getU());

	}

	private static double norm(JavaSparkContext sc, final JavaRDD<org.apache.spark.mllib.linalg.Vector> vectors,
			final int nRows, final int nCols, final int nPCs, final int subsample, final int q, final Vector meanVector,
			final Matrix centralC) {
		/************************** SSVD PART *****************************/

		/**
		 * Sketch dimension ,S=s Sketched matrix, B=A*S; QR
		 * decomposition, Q=qr(B); SV decomposition, [~,s,V]=svd(Q);
		 */

		// initialize & broadcast a random seed
		org.apache.spark.mllib.linalg.Matrix GaussianRandomMatrix = org.apache.spark.mllib.linalg.Matrices.randn(nCols,
				nPCs + subsample, new SecureRandom());
		// PCAUtils.printMatrixToFile(GaussianRandomMatrix, OutputFormat.DENSE,
		// outputPath+File.separator+"Seed");
		Matrix B = PCAUtils.convertSparkToMahoutMatrix(GaussianRandomMatrix);

		Matrix V = new org.apache.mahout.math.SingularValueDecomposition(centralC).getU();

		org.apache.mahout.math.SingularValueDecomposition SVD = null;

		double S = 0;

		for (int iter = 0; iter < q; iter++) {
			// V'*omega
			final Matrix VtSeed = V.transpose().times(B);
			// V*V'*omega
			final Matrix VVtSeed = V.times(VtSeed);
			// omega-V*V'*omega
			Matrix Seed = B.minus(VVtSeed);

			Vector seedMu = Seed.transpose().times(meanVector);
			final Broadcast<Vector> brSeedMu = sc.broadcast(seedMu);
			// System.out.println(brSeedMu.value().getQuick(5));

			final Broadcast<Matrix> seed = sc.broadcast(Seed);
			
			final int s= nPCs+subsample;
			final Accumulator<double[][]> matrixAccumYtY = sc.accumulator(new double[s][s],
					new MatrixAccumulatorParam());
			final Accumulator<double[]> matrixAccumY = sc.accumulator(new double[s], new VectorAccumulatorParam());

			/*
			 * Initialize the output matrices and vectors once in order to avoid
			 * generating massive intermediate data in the workers
			 */
			final double[][] resArrayYtY = new double[s][s];
			final double[] resArrayY = new double[s];

			/*
			 * Used to sum the vectors in one partition.
			 */
			final double[][] internalSumYtY = new double[s][s];
			final double[] internalSumY = new double[s];

			vectors.foreachPartition(new VoidFunction<Iterator<org.apache.spark.mllib.linalg.Vector>>() {

				public void call(Iterator<org.apache.spark.mllib.linalg.Vector> arg0) throws Exception {
					org.apache.spark.mllib.linalg.Vector yi;
					while (arg0.hasNext()) {
						yi = arg0.next();

						/*
						 * Perform in-memory matrix multiplication xi = yi' *
						 * Y2X
						 */
						PCAUtils.sparseVectorTimesMatrix(yi, seed.value(), resArrayY);

						// get only the sparse indices
					
						PCAUtils.outerProductArrayInput(resArrayY, brSeedMu.value(), resArrayY,
								brSeedMu.value(), resArrayYtY);
						int i, j;

						// add the sparse indices only
						
						for (i = 0; i < s; i++) {
							internalSumY[i] += resArrayY[i];
							for (j = 0; j < s; j++) {
								internalSumYtY[i][j] += resArrayYtY[i][j];
								resArrayYtY[i][j] = 0; // reset it
							}

						}
					}
					matrixAccumY.add(internalSumY);
					matrixAccumYtY.add(internalSumYtY);
				}

			});// end X'X and Y'X Job

			/*
			 * Get the values of the accumulators.
			 */
			Matrix centralYtY = new DenseMatrix(matrixAccumYtY.value());
			Vector centralSumY = new DenseVector(matrixAccumY.value());

			/*
			 * Mi = (Yi-Ym)' x (Xi-Xm) = Yi' x (Xi-Xm) - Ym' x (Xi-Xm)
			 * 
			 * M = Sum(Mi) = Sum(Yi' x (Xi-Xm)) - Ym' x (Sum(Xi)-N*Xm)
			 * 
			 * The first part is done in the previous job and the second in the
			 * following method
			 */
			centralYtY = PCAUtils.updateXtXAndYtx(centralYtY, centralSumY, seedMu, seedMu, nRows);

			
			Matrix R=new org.apache.mahout.math.CholeskyDecomposition(centralYtY,false).getL().transpose();			
			

			R = PCAUtils.inv(R);
			// omega-V*V'*omega
			Seed = Seed.times(R);

			seedMu = Seed.transpose().times(meanVector);
			final Broadcast<Vector> brSeedMu_R = sc.broadcast(seedMu);
			// System.out.println(brSeedMu.value().getQuick(5));

			final Broadcast<Matrix> seed_R = sc.broadcast(Seed);

			

			final Accumulator<double[][]> matrixAccumAtQ = sc.accumulator(new double[nCols][s],
					new MatrixAccumulatorParam());
			final Accumulator<double[]> matrixAccumQ = sc.accumulator(new double[s], new VectorAccumulatorParam());

			/*
			 * Initialize the output matrices and vectors once in order to avoid
			 * generating massive intermediate data in the workers
			 */
			final double[][] resArrayAtQ = new double[nCols][s];
			final double[] resArrayQ = new double[s];

			/*
			 * Used to sum the vectors in one partition.
			 */
			final double[][] internalSumAtQ = new double[nCols][s];
			final double[] internalSumQ = new double[s];

			vectors.foreachPartition(new VoidFunction<Iterator<org.apache.spark.mllib.linalg.Vector>>() {

				public void call(Iterator<org.apache.spark.mllib.linalg.Vector> arg0) throws Exception {
					org.apache.spark.mllib.linalg.Vector yi;
					while (arg0.hasNext()) {
						yi = arg0.next();

						/*
						 * Perform in-memory matrix multiplication xi = yi' *
						 * Y2X
						 */
						PCAUtils.sparseVectorTimesMatrix(yi, seed_R.value(), resArrayQ);

						// get only the sparse indices
						int[] indices = ((SparseVector) yi).indices();

						PCAUtils.outerProductWithIndices(yi, brSeedMu.value(), resArrayQ, brSeedMu_R.value(),
								resArrayAtQ, indices);
						
						int i, j, rowIndexYtX;

						// add the sparse indices only
						for (i = 0; i < indices.length; i++) {
							rowIndexYtX = indices[i];
							for (j = 0; j < s; j++) {
								internalSumAtQ[rowIndexYtX][j] += resArrayAtQ[rowIndexYtX][j];
								resArrayAtQ[rowIndexYtX][j] = 0; // reset it
							}

						}
						for (i = 0; i < s; i++) {
							internalSumQ[i] += resArrayQ[i];
							

						}
					}
					matrixAccumQ.add(internalSumQ);
					matrixAccumAtQ.add(internalSumAtQ);
				}

			});// end X'X and Y'X Job

			/*
			 * Get the values of the accumulators.
			 */
			Matrix centralAtQ = new DenseMatrix(matrixAccumAtQ.value());
			Vector centralSumQ = new DenseVector(matrixAccumQ.value());
			
			centralAtQ = PCAUtils.updateXtXAndYtx(centralAtQ, centralSumQ, meanVector, seedMu, nRows);
			
			centralAtQ=centralAtQ.transpose();
			
			final Matrix QtAV = centralAtQ.times(V);
			final Matrix QtAVVt = QtAV.times(V.transpose());
			B = centralAtQ.minus(QtAVVt);

			SVD = new org.apache.mahout.math.SingularValueDecomposition(B);

			B = B.transpose();

			Double newS = SVD.getS().getQuick(nPCs - 1, nPCs - 1);
			newS = Math.round(newS * 10000.0) / 10000.0;
			if(newS==S) break;
			else S=newS;
			System.out.println(S);
		}

		return S;

	}

	private static void printLogMessage(String argName) {
		log.error("Missing arguments -D" + argName);
		log.info(
				"Usage: -Di=<path/to/input/matrix> -Do=<path/to/outputfolder> -Drows=<number of rows> -Dcols=<number of columns> -Dpcs=<number of principal components> [-DerrSampleRate=<Error sampling rate>] [-DmaxIter=<max iterations>] [-DoutFmt=<output format>] [-DcalculateError=<0/1 (compute projected matrix or not)>]");
	}
}