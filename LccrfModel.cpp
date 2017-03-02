#include <unordered_set>
#include <boost/filesystem.hpp>
#include <Eigen/Dense>
#include "LccrfModel.h"
#include "util.h"

const char* LccrfModel::kOrderOption = "lccrf.order";

namespace {
	const int kTokenBufferCount = 100;
	const int kLabelBufferCount = 40;
}

LccrfModel::LccrfModel()
{
	optionsdesc_.add_options()
		(kOrderOption, boost::program_options::value<int>()->default_value(1), "the order of markov chain");
	maxunigramid_ = maxbigramid_ = maxlabelid_ = 0;
	uniweightsize_ = 0;
	markovorder_ = 0;
}

LccrfModel::~LccrfModel()
{

}

void LccrfModel::InitFromCmd(int argc, const char * argv[])
{
	auto vm = ParseArgs(argc, argv, optionsdesc_, true);
	markovorder_ = vm[kOrderOption].as<int>();

}

void LccrfModel::InitFromData(DataIterator & iterator)
{
	auto dat = iterator.GetAllData();
	auto label = iterator.GetAllLabel();
	maxunigramid_ = dat.GetMaxUnigramFeatureId();
	maxbigramid_ = dat.GetMaxBigramFeatureId();
	maxlabelid_ = label.GetMaxLabelId();

	modelsize_ = 0;
	modelsize_ += (maxunigramid_ + 1)*(maxlabelid_ + 1);
	modelsize_ += (maxbigramid_ + 1)*(maxlabelid_ + 1)*(maxlabelid_ + 1);

	if (modelsize_ < 1) {
		BOOST_LOG_TRIVIAL(fatal) << "Model size error";
		return;
	}

	param_.reset(new DenseVector(modelsize_));
	if (param_.get() == nullptr) {
		BOOST_LOG_TRIVIAL(fatal) << "Failed to allocate model parameters";
		return;
	}

	uniweightsize_ = (maxunigramid_ + 1)*(maxlabelid_ + 1);
	param_->setZero();
	sparseunikeys_.reserve(maxunigramid_ + 1 > 100000 ? 100000 : (maxunigramid_ + 1));
	sparsebikeys_.reserve(maxbigramid_ + 1);
	sparsekeys_.reserve(modelsize_);
	sparseparam_.reset(new DenseVector(modelsize_));
	if (!sparseparam_.get()) {
		BOOST_LOG_TRIVIAL(fatal) << "Failed to allocate memory for sparse param";
		return;
	}
}

bool LccrfModel::LoadModel(std::string model)
{
	namespace fs = boost::filesystem;
	fs::path srcpath(model);

	if (!fs::exists(srcpath)) {
		BOOST_LOG_TRIVIAL(fatal) << "model path do not exist " << model;
		return false;
	}

	std::fstream src(model, std::ios_base::in | std::ios_base::binary);
	if (!src.is_open()) {
		BOOST_LOG_TRIVIAL(fatal) << "failed to open " << model;
		return false;
	}

	BinaryFileHandler reader(src);
	reader.ReadSizeT(maxunigramid_);
	reader.ReadSizeT(maxbigramid_);
	reader.ReadSizeT(maxlabelid_);
	reader.ReadSizeT(modelsize_);

	param_.reset(new DenseVector(modelsize_));
	double temp;
	for (int i = 0; i < modelsize_; ++i) {
		reader.ReadFloat(param_->coeffRef(i));
	}

	return true;
}

bool LccrfModel::SaveModel(std::string model)
{

	std::fstream sink(model, std::ios_base::out | std::ios_base::binary);
	if (!sink.is_open()) {
		BOOST_LOG_TRIVIAL(fatal) << "Failed to open " << model;
		return false;
	}

	BinaryFileHandler writer(sink);

	writer.WriteSizeT(maxunigramid_);
	writer.WriteSizeT(maxbigramid_);
	writer.WriteSizeT(maxlabelid_);
	writer.WriteSizeT(modelsize_);

	for (int i = 0; i < modelsize_; ++i) {
		writer.WriteFloat(param_->coeffRef(i));
	}

	return true;
}

double LccrfModel::Learn(LccrfSamples& samples, LccrfLabels& labels, SparseVector& grad)
{
	namespace signal = boost::signals2::detail;

	if (grad.size() < modelsize_)
		grad.resize(modelsize_);

	double avgloss = 0.0;
	int labelcount = maxlabelid_ + 1;
	int index = 0;
	sparseunikeys_.clear();
	sparsebikeys_.clear();

#ifdef _DEBUG
	std::unordered_set<int> uniqunifeats, uniqbifeats;
#endif // _DEBUG

	for (int i = 0; i < samples.NumSamples(); ++i) {

		DataSamples& unigramfeature = *(samples.UnigramFeature()[i]);
		DataSamples& bigramfeature = *(samples.BigramFeature()[i]);
		LabelVector& label = *(labels.Labels()[i]);

#ifdef _DEBUG
		BOOST_LOG_TRIVIAL(info) << "unigram count " << unigramfeature.nonZeros();
		BOOST_LOG_TRIVIAL(info) << "bigram count " << bigramfeature.nonZeros();
#endif // _DEBUG

		int wordcount = unigramfeature.rows();
		int uniscoresize = wordcount * labelcount;
		int biscoresize = (wordcount - 1) * labelcount * labelcount;

		signal::auto_buffer<double, signal::store_n_objects<kTokenBufferCount*kLabelBufferCount>> nodeprobuffer(uniscoresize);
		signal::auto_buffer<double, signal::store_n_objects<kTokenBufferCount*kLabelBufferCount*kLabelBufferCount>> edgeprobuffer(biscoresize);

		Eigen::MatrixXd nodeprob = Eigen::Map<Eigen::MatrixXd>(nodeprobuffer.data(), labelcount, wordcount);
		Eigen::MatrixXd edgeprob = Eigen::Map<Eigen::MatrixXd>(edgeprobuffer.data(), labelcount*labelcount, wordcount - 1);

		avgloss += GetNodeAndEdgeProb(unigramfeature, bigramfeature, nodeprob, edgeprob, label);

		for (int wordpos = 0; wordpos < wordcount; ++wordpos) {
			for (DataSamples::InnerIterator iter(unigramfeature, wordpos); iter; ++iter) {

#ifdef _DEBUG
				uniqunifeats.insert(iter.col());
#endif // _DEBUG
				if (!sparseunikeys_.count(iter.col())) {
					sparseunikeys_.insert(iter.col());
					index = GetUnigramFeatureIndex(iter.col(), 0);
					for (int i = 0; i <= maxlabelid_; ++i) {
						sparseparam_->coeffRef(index + i) = 0;
					}
				}

				index = GetUnigramFeatureIndex(iter.col(), label.coeff(wordpos));
				sparseparam_->coeffRef(index) -= iter.value();

				for (int unilabel = 0; unilabel <= maxlabelid_; ++unilabel) {
					index = GetUnigramFeatureIndex(iter.col(), unilabel);
					sparseparam_->coeffRef(index) += nodeprob(unilabel, wordpos)*iter.value();
				}
			}

			if (wordpos < wordcount - 1) {
				for (DataSamples::InnerIterator iter(bigramfeature, wordpos); iter; ++iter) {

#ifdef _DEBUG
					uniqbifeats.insert(iter.col());
#endif // _DEBUG
					if (!sparsebikeys_.count(iter.col())) {
						sparsebikeys_.insert(iter.col());
						index = GetBigramFeatureIndex(iter.col(), 0, 0);
						for (int i = 0; i < (maxlabelid_ + 1)*(maxlabelid_ + 1); ++i) {
							sparseparam_->coeffRef(index + i) = 0;
						}
					}


					int index = GetBigramFeatureIndex(iter.col(), label.coeff(wordpos), label.coeff(wordpos + 1));
					sparseparam_->coeffRef(index) -= iter.value();

					for (int fromlabel = 0; fromlabel <= maxlabelid_; ++fromlabel) {
						for (int tolabel = 0; tolabel <= maxlabelid_; ++tolabel) {
							index = GetBigramFeatureIndex(iter.col(), fromlabel, tolabel);
							int edgeindex = fromlabel * (maxlabelid_ + 1) + tolabel;
							sparseparam_->coeffRef(index) += edgeprob(edgeindex, wordpos) * iter.value();
						}
					}
				}
			}
		}
	}

	int nonzeros = sparseunikeys_.size()*(maxlabelid_ + 1) + sparsebikeys_.size()*(maxlabelid_ + 1)*(maxlabelid_ + 1);
	grad.reserve(nonzeros);
	grad.setZero();

	sparsekeys_.resize(sparseunikeys_.size());
	index = 0;
	for (std::unordered_set<size_t>::iterator iter = sparseunikeys_.begin();
		iter != sparseunikeys_.end(); ++iter, ++index) {
		sparsekeys_[index] = *iter;
	}

	std::sort(sparsekeys_.begin(), sparsekeys_.end());
#ifdef _DEBUG
	BOOST_LOG_TRIVIAL(info) << "key count         :" << sparsekeys_.size();
	BOOST_LOG_TRIVIAL(info) << "uniq unigram keys :" << uniqunifeats.size();
	BOOST_LOG_TRIVIAL(info) << "uniq bigram keys  :" << uniqbifeats.size();
#endif // _DEBUG

	int* innerindex = grad.innerIndexPtr();
	float* innervals = grad.valuePtr();
	for (int i = 0; i < sparsekeys_.size(); ++i) {
		index = GetUnigramFeatureIndex(sparsekeys_[i], 0);
		for (int offset = 0; offset <= maxlabelid_; ++offset, ++innerindex, ++innervals) {
			*innerindex = (index + offset);
			*innervals = sparseparam_->coeff(index + offset);
		}
	}

	sparsekeys_.resize(sparsebikeys_.size());
	index = 0;
	for (std::unordered_set<size_t>::iterator iter = sparsebikeys_.begin();
		iter != sparsebikeys_.end(); ++iter, ++index) {
		sparsekeys_[index] = *iter;
	}
	std::sort(sparsekeys_.begin(), sparsekeys_.end());
	for (int i = 0; i < sparsekeys_.size(); ++i) {
		index = GetBigramFeatureIndex(sparsekeys_[i], 0, 0);
		for (int offset = 0; offset < (maxlabelid_ + 1)*(maxlabelid_ + 1); ++offset, ++innerindex, ++innervals) {
			*innerindex = (index + offset);
			*innervals = sparseparam_->coeff(index + offset);
		}
	}

	grad.resizeNonZeros(nonzeros);
	grad /= samples.NumSamples();
	avgloss /= samples.NumSamples();
	return avgloss;
}

double LccrfModel::Learn(LccrfSamples& samples, LccrfLabels& labels, DenseVector& grad)
{
	namespace signal = boost::signals2::detail;
	double avgloss = 0.0;
	int labelcount = maxlabelid_ + 1;

	grad.resize(modelsize_);
	grad.setZero();

	for (int i = 0; i < samples.NumSamples(); ++i) {

		DataSamples& unigramfeature = *(samples.UnigramFeature()[i]);
		DataSamples& bigramfeature = *(samples.BigramFeature()[i]);
		LabelVector& label = *(labels.Labels()[i]);

		int wordcount = unigramfeature.rows();
		int uniscoresize = wordcount * labelcount;
		int biscoresize = (wordcount - 1) * labelcount * labelcount;

		signal::auto_buffer<double, signal::store_n_objects<kTokenBufferCount*kLabelBufferCount>> nodeprobuffer(uniscoresize);
		signal::auto_buffer<double, signal::store_n_objects<kTokenBufferCount*kLabelBufferCount*kLabelBufferCount>> edgeprobuffer(biscoresize);
		Eigen::MatrixXd nodeprob = Eigen::Map<Eigen::MatrixXd>(nodeprobuffer.data(), labelcount, wordcount);
		Eigen::MatrixXd edgeprob = Eigen::Map<Eigen::MatrixXd>(edgeprobuffer.data(), labelcount*labelcount, wordcount - 1);

		avgloss += GetNodeAndEdgeProb(unigramfeature, bigramfeature, nodeprob, edgeprob, label);

		for (int wordpos = 0; wordpos < wordcount; ++wordpos) {
			for (DataSamples::InnerIterator iter(unigramfeature, wordpos); iter; ++iter) {
				int index = GetUnigramFeatureIndex(iter.col(), label.coeff(wordpos));
				grad.coeffRef(index) -= iter.value();
				for (int unilabel = 0; unilabel <= maxlabelid_; ++unilabel) {
					index = GetUnigramFeatureIndex(iter.col(), unilabel);
					grad.coeffRef(index) += nodeprob(unilabel, wordpos)*iter.value();
				}
			}

			if (wordpos < wordcount - 1) {
				for (DataSamples::InnerIterator iter(bigramfeature, wordpos); iter; ++iter) {
					int index = GetBigramFeatureIndex(iter.col(), label.coeff(wordpos), label.coeff(wordpos + 1));
					grad.coeffRef(index) -= iter.value();
					for (int fromlabel = 0; fromlabel <= maxlabelid_; ++fromlabel) {
						for (int tolabel = 0; tolabel <= maxlabelid_; ++tolabel) {
							index = GetBigramFeatureIndex(iter.col(), fromlabel, tolabel);
							int edgeindex = fromlabel * (maxlabelid_ + 1) + tolabel;
							grad.coeffRef(index) += edgeprob(edgeindex, wordpos)*iter.value();
						}
					}
				}
			}
		}
	}

	grad /= samples.NumSamples();
	avgloss /= samples.NumSamples();
	return avgloss;
}

void LccrfModel::Inference(LccrfSamples& samples, LccrfLabels& labels)
{
	namespace signal = boost::signals2::detail;
	labels.Labels().resize(samples.NumSamples());

	for (int i = 0; i < samples.NumSamples(); ++i) {

		DataSamples& unigramfeature = *(samples.UnigramFeature(i));
		DataSamples& bigramfeature = *(samples.BigramFeature(i));

		int labelcount = maxlabelid_ + 1;
		int wordcount = unigramfeature.rows();
		int uniscoresize = wordcount * labelcount;
		int biscoresize = (wordcount - 1) * labelcount * labelcount;

		labels.Labels(i).reset(new LabelVector(wordcount));

		signal::auto_buffer<double, signal::store_n_objects<kTokenBufferCount*kLabelBufferCount>> uniscorebuffer(uniscoresize);
		signal::auto_buffer<double, signal::store_n_objects<kTokenBufferCount*kLabelBufferCount*kLabelBufferCount>> biscorebuffer(biscoresize);

		signal::auto_buffer<double, signal::store_n_objects<kTokenBufferCount*kLabelBufferCount>> alphabuffer(uniscoresize);
		signal::auto_buffer<double, signal::store_n_objects<kTokenBufferCount*kLabelBufferCount>> betabuffer(uniscoresize);

		Eigen::MatrixXd uniscore = Eigen::Map<Eigen::MatrixXd>(uniscorebuffer.data(), labelcount, wordcount);
		Eigen::MatrixXd biscore = Eigen::Map<Eigen::MatrixXd>(biscorebuffer.data(), labelcount*labelcount, wordcount - 1);

		CalculateUnigramScore(unigramfeature, uniscore);
		CalculateBigramScore(bigramfeature, biscore);

		Viterbi1Best(uniscore, biscore, wordcount, *(labels.Labels(i)));
	}
}

double LccrfModel::Evaluate(LccrfSamples& samples, LccrfLabels& labels, std::string& summary)
{
	namespace signal = boost::signals2::detail;

	double logli = 0;
	float totalcounts = 0, rightcount = 0;
	for (int i = 0; i < samples.NumSamples(); ++i) {

		DataSamples& unigramfeature = *(samples.UnigramFeature(i));
		DataSamples& bigramfeature = *(samples.BigramFeature(i));
		LabelVector& label = *(labels.Labels(i));

		int labelcount = maxlabelid_ + 1;
		int wordcount = unigramfeature.rows();
		int uniscoresize = wordcount * labelcount;
		int biscoresize = (wordcount - 1) * labelcount * labelcount;

		signal::auto_buffer<double, signal::store_n_objects<kTokenBufferCount*kLabelBufferCount>> uniscorebuffer(uniscoresize);
		signal::auto_buffer<double, signal::store_n_objects<kTokenBufferCount*kLabelBufferCount*kLabelBufferCount>> biscorebuffer(biscoresize);

		signal::auto_buffer<double, signal::store_n_objects<kTokenBufferCount*kLabelBufferCount>> alphabuffer(uniscoresize);
		signal::auto_buffer<double, signal::store_n_objects<kTokenBufferCount*kLabelBufferCount>> betabuffer(uniscoresize);

		signal::auto_buffer<double, signal::store_n_objects<kLabelBufferCount>> partionbuffer(maxlabelid_ + 1);

		Eigen::MatrixXd uniscore = Eigen::Map<Eigen::MatrixXd>(uniscorebuffer.data(), labelcount, wordcount);
		Eigen::MatrixXd biscore = Eigen::Map<Eigen::MatrixXd>(biscorebuffer.data(), labelcount*labelcount, wordcount - 1);

		Eigen::MatrixXd alpha = Eigen::Map<Eigen::MatrixXd>(alphabuffer.data(), labelcount, wordcount);
		Eigen::MatrixXd beta = Eigen::Map<Eigen::MatrixXd>(betabuffer.data(), labelcount, wordcount);

		Eigen::VectorXd lastalpha = Eigen::Map<Eigen::VectorXd>(partionbuffer.data(), maxlabelid_ + 1);
		LabelVector predictlabels;

		CalculateUnigramScore(unigramfeature, uniscore);
		CalculateBigramScore(bigramfeature, biscore);

		ForwardPass(uniscore, biscore, wordcount, alpha);
		Viterbi1Best(uniscore, biscore, wordcount, predictlabels);

		lastalpha = alpha.col(wordcount - 1);
		logli += LogLikelihood(lastalpha, uniscore, biscore, label, wordcount);
		totalcounts += wordcount;
		for (int wordpos = 0; wordpos < wordcount; ++wordpos) {
			if (predictlabels.coeff(wordpos) == label.coeff(wordpos)) {
				rightcount += 1;
			}
		}
	}

	logli /= samples.NumSamples();
	std::stringstream strout;
	strout << "Average log-likelihood " << logli << " over " << samples.NumSamples() << " samples" << std::endl;
	strout << "Accuracy rate " << (rightcount / totalcounts) * 100;
	summary = strout.str();
	return logli;
}

void LccrfModel::CalculateUnigramScore(DataSamples& sample, DenseMatrix& val)
{
	val.setZero();
	for (int wordpos = 0; wordpos < sample.rows(); wordpos++) {		
		for (int labelid = 0; labelid <= maxlabelid_; ++labelid) {
			for (DataSamples::InnerIterator iter(sample, wordpos); iter; ++iter) {
				val(labelid, wordpos) += iter.value() * GetUnigramWeight(iter.col(), labelid);
			}
		}
	}

#ifdef _DEBUG
	BOOST_ASSERT(!val.hasNaN());
#endif // _DEBUG

}

void LccrfModel::CalculateBigramScore(DataSamples& sample, DenseMatrix& val)
{
	val.setZero();

#ifdef _DEBUG
	BOOST_ASSERT_MSG(val.rows() == (maxlabelid_ + 1)*(maxlabelid_ + 1), "Error columns for bigram weights");
	BOOST_ASSERT_MSG(val.cols() == (sample.rows() - 1), "error rows for bigram feature");
#endif // !_DEBUG


	for (int wordpos = 0; wordpos < sample.rows() - 1; ++wordpos) {
		for (int fromlabel = 0; fromlabel <= maxlabelid_; ++fromlabel) {
			for (int tolabel = 0; tolabel <= maxlabelid_; ++tolabel) {
				int index = fromlabel * (maxlabelid_ + 1) + tolabel;
				for (DataSamples::InnerIterator iter(sample, wordpos); iter; ++iter) {
					val(index, wordpos) += iter.value() * GetBigramWeight(iter.col(), fromlabel, tolabel);
				}
			}
		}
	}

#ifdef _DEBUG
	BOOST_ASSERT_MSG(!val.hasNaN(), "Has Nan when calculate Bigram scores");
#endif // _DEBUG

}

void LccrfModel::ForwardPass(DenseMatrix& node, DenseMatrix& edge, int wordcount, DenseMatrix& alpha)
{
	namespace signal = boost::signals2::detail;

	alpha.setZero();
	alpha.col(0) = node.col(0);
	signal::auto_buffer<double, signal::store_n_objects<kLabelBufferCount>> transbuffer(maxlabelid_ + 1);
	Eigen::VectorXd trans = Eigen::Map<Eigen::VectorXd>(transbuffer.data(), maxlabelid_ + 1);

	for (int wordpos = 1; wordpos < wordcount; ++wordpos) {

		alpha.col(wordpos) = node.col(wordpos);

		for (int label = 0; label <= maxlabelid_; ++label) {
			trans.setZero();
			for (int fromlabel = 0; fromlabel <= maxlabelid_; ++fromlabel) {
				int index = fromlabel*(maxlabelid_ + 1) + label;
				trans[fromlabel] = edge(index, wordpos - 1);
			}

			trans += alpha.col(wordpos - 1);
			alpha(label, wordpos) += LogSumExp(trans);
		}
	}

#ifdef _DEBUG
	BOOST_ASSERT_MSG(!alpha.hasNaN(), "Get Nan when compute Forward Pass ");
#endif // _DEBUG

}

void LccrfModel::BackwardPass(DenseMatrix& node, DenseMatrix& edge, int wordcount, DenseMatrix& beta)
{
	namespace signal = boost::signals2::detail;
	signal::auto_buffer<double, signal::store_n_objects<kLabelBufferCount>> transbuffer(maxlabelid_ + 1);
	Eigen::VectorXd trans = Eigen::Map<Eigen::VectorXd>(transbuffer.data(), maxlabelid_ + 1);

	beta.setZero();
	for (int wordpos = wordcount - 2; wordpos >= 0; --wordpos) {
		for (int label = 0; label <= maxlabelid_; ++label) {
			trans.setZero();
			trans = beta.col(wordpos + 1);
			for (int tolabel = 0; tolabel <= maxlabelid_; ++tolabel) {
				int index = label * (maxlabelid_ + 1) + tolabel;
				trans.coeffRef(tolabel) += edge(index, wordpos);
			}
			trans += node.col(wordpos + 1);

			beta.coeffRef(label, wordpos) = LogSumExp(trans);
		}
	}

#ifdef _DEBUG
	BOOST_ASSERT_MSG(!beta.hasNaN(), "Get Nan When compute Backward pass");
#endif // _DEBUG

}

void LccrfModel::NodeProb(DenseMatrix& alpha, DenseMatrix& beta, int wordcount, DenseMatrix& nodeprob)
{
	namespace signal = boost::signals2::detail;
	signal::auto_buffer<double, signal::store_n_objects<kLabelBufferCount>> probuffer(maxlabelid_ + 1);
	Eigen::VectorXd prob = Eigen::Map<Eigen::VectorXd>(probuffer.data(), maxlabelid_ + 1);

	nodeprob.setZero();
	for (int wordpos = 0; wordpos < wordcount; ++wordpos) {
		prob.setZero();
		prob = alpha.col(wordpos) + beta.col(wordpos);
		double m = prob.maxCoeff();
		prob = prob.array() - m;
		double total = prob.array().exp().sum();
		nodeprob.col(wordpos) = prob.array().exp() / total;
	}
}

void LccrfModel::EdgeProb(DenseMatrix& alpha, DenseMatrix& beta, DenseMatrix& node, DenseMatrix& edge, int wordcount, DenseMatrix& edgeprob) {
	namespace signal = boost::signals2::detail;
	signal::auto_buffer<double, signal::store_n_objects<kLabelBufferCount*kLabelBufferCount>> edgeprobuffer((maxlabelid_ + 1)*(maxlabelid_ + 1));
	Eigen::VectorXd prob = Eigen::Map<Eigen::VectorXd>(edgeprobuffer.data(), (maxlabelid_ + 1)*(maxlabelid_ + 1));

	edgeprob.setZero();

	for (int wordpos = 0; wordpos < wordcount - 1; ++wordpos) {
		int nextpos = wordpos + 1;
		prob.setZero();
		for (int fromlabel = 0; fromlabel <= maxlabelid_; ++fromlabel) {
			for (int tolabel = 0; tolabel <= maxlabelid_; ++tolabel) {
				int index = fromlabel * (maxlabelid_ + 1) + tolabel;
				prob.coeffRef(index) += alpha(fromlabel, wordpos) + beta(tolabel, nextpos) + edge(index, wordpos) + node(tolabel, nextpos);
			}
		}

		double m = prob.maxCoeff();
		double total = 0;
		prob = prob.array() - m;
		total = prob.array().exp().sum();
		edgeprob.col(wordpos) = prob.array().exp() / total;
	}
}

void LccrfModel::Viterbi1Best(DenseMatrix& node, DenseMatrix& edge, int wordcount, LabelVector& path)
{
	namespace signal = boost::signals2::detail;
	signal::auto_buffer<int, signal::store_n_objects<kTokenBufferCount*kLabelBufferCount>> trackbuffer(wordcount*(maxlabelid_ + 1));
	signal::auto_buffer<double, signal::store_n_objects<kTokenBufferCount*kLabelBufferCount>> maxpathbuffer(wordcount*(maxlabelid_ + 1));

	Eigen::MatrixXi track = Eigen::Map<Eigen::MatrixXi>(trackbuffer.data(), maxlabelid_ + 1, wordcount);
	Eigen::MatrixXd maxpath = Eigen::Map<Eigen::MatrixXd>(maxpathbuffer.data(), maxlabelid_ + 1, wordcount);

	track.setOnes();
	track *= -1;
	maxpath.setZero();
	maxpath.col(0) = node.col(0);
	path.resize(wordcount);

	for (int wordpos = 1; wordpos < wordcount; ++wordpos) {
		for (int label = 0; label <= maxlabelid_; ++label) {
			int labelid = -1;
			double maxscore = std::numeric_limits<double>::min();
			for (int fromlabel = 0; fromlabel <= maxlabelid_; ++fromlabel) {
				int index = fromlabel * (maxlabelid_ + 1) + label;
				double score = maxpath(fromlabel, wordpos - 1) + edge(index, wordpos - 1) + node(label, wordpos);
				if (score > maxscore) {
					labelid = fromlabel;
					maxscore = score;
				}
			}

			maxpath(label, wordpos) = maxscore;
			track(label, wordpos) = labelid;
		}
	}

#ifdef _DEBUG
	BOOST_ASSERT_MSG(!maxpath.hasNaN(), "Max path has NaN");
#endif // _DEBUG


	// backtrack stage
	int bestpath = 0;
	double maxscore = std::numeric_limits<double>::min();
	for (int label = 0; label <= maxlabelid_; ++label) {
		if (maxpath(label, wordcount - 1) > maxscore) {
			bestpath = label;
		}
	}

	path.coeffRef(wordcount - 1) = bestpath;
	for (int wordpos = wordcount - 1; wordpos > 0; --wordpos) {
		path.coeffRef(wordpos - 1) = track.coeff(bestpath, wordpos);
		bestpath = path.coeff(wordpos - 1);
	}
}

double LccrfModel::GetNodeAndEdgeProb(DataSamples & unigramfeature, DataSamples & bigramfeature,
	DenseMatrix & nodeprob, DenseMatrix & edgeprob,
	LabelVector& labels)
{
	namespace signal = boost::signals2::detail;

	int labelcount = maxlabelid_ + 1;
	int wordcount = unigramfeature.rows();
	int uniscoresize = wordcount * labelcount;
	int biscoresize = (wordcount - 1) * labelcount * labelcount;

	signal::auto_buffer<double, signal::store_n_objects<kTokenBufferCount*kLabelBufferCount>> uniscorebuffer(uniscoresize);
	signal::auto_buffer<double, signal::store_n_objects<kTokenBufferCount*kLabelBufferCount*kLabelBufferCount>> biscorebuffer(biscoresize);
	signal::auto_buffer<double, signal::store_n_objects<kLabelBufferCount>> partionbuffer(maxlabelid_ + 1);

	signal::auto_buffer<double, signal::store_n_objects<kTokenBufferCount*kLabelBufferCount>> alphabuffer(uniscoresize);
	signal::auto_buffer<double, signal::store_n_objects<kTokenBufferCount*kLabelBufferCount>> betabuffer(uniscoresize);

	Eigen::MatrixXd uniscore = Eigen::Map<Eigen::MatrixXd>(uniscorebuffer.data(), labelcount, wordcount);
	Eigen::MatrixXd biscore = Eigen::Map<Eigen::MatrixXd>(biscorebuffer.data(), labelcount*labelcount, wordcount - 1);

	Eigen::MatrixXd alpha = Eigen::Map<Eigen::MatrixXd>(alphabuffer.data(), labelcount, wordcount);
	Eigen::MatrixXd beta = Eigen::Map<Eigen::MatrixXd>(betabuffer.data(), labelcount, wordcount);

	Eigen::VectorXd lastalpha = Eigen::Map<Eigen::VectorXd>(partionbuffer.data(), maxlabelid_ + 1);

	CalculateUnigramScore(unigramfeature, uniscore);
	CalculateBigramScore(bigramfeature, biscore);

	ForwardPass(uniscore, biscore, wordcount, alpha);
	BackwardPass(uniscore, biscore, wordcount, beta);

	NodeProb(alpha, beta, wordcount, nodeprob);
	EdgeProb(alpha, beta, uniscore, biscore, wordcount, edgeprob);

	lastalpha = alpha.col(wordcount - 1);
	return LogLikelihood(lastalpha, uniscore, biscore, labels, wordcount);
}

double LccrfModel::LogLikelihood(Eigen::VectorXd& lastalpha, DenseMatrix& node, DenseMatrix& edge, LabelVector& labels, int wordcount)
{
	double logpartion = LogSumExp(lastalpha);

	int prevlabel = labels.coeff(0);
	logpartion -= node.coeff(prevlabel, 0);
	for (int wordpos = 1; wordpos < wordcount; ++wordpos) {
		int labelid = labels.coeff(wordpos);
		int index = prevlabel * (maxlabelid_ + 1) + labelid;
		logpartion -= edge.coeff(index, wordpos - 1);
		logpartion -= edge.coeff(labelid, wordpos);
	}
	return logpartion;
}

double LccrfModel::LogSumExp(Eigen::VectorXd & v)
{
	double max = v.maxCoeff();
	return max + std::log((v.array() - max).exp().sum());
}

