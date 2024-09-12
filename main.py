from src.utils_data import read_and_transform_data
from src.utils_results import results_reports
from src.poolings import PoolingStrategy
from src.prediction_strategies import PredictionStrategy
from src.word_sense_detector import WordSenseDetector
from src.udpipe_model import UDPipeModel
import warnings
warnings.filterwarnings('ignore')

data = read_and_transform_data('datasets_pre_defined/sum_14_final.jsonlines', homonym=True)

udpipe_model = UDPipeModel("datasets_pre_defined/20180506.uk.mova-institute.udpipe")
word_sense_detector = WordSenseDetector(
    pretrained_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    udpipe_model=udpipe_model,
    evaluation_dataset=data.head(100),
    pooling_strategy=PoolingStrategy.mean_pooling,
    prediction_strategy=PredictionStrategy.max_sim_across_all_examples
)
evaluation_dataset_pd = word_sense_detector.run()

results_reports(evaluation_dataset_pd, udpipe_model)
