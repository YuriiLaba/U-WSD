from src.utils_data import read_and_transform_data
from src.utils_results import results_reports
from src.word_sense_detector import WordSenseDetector
from src.udpipe_model import UDPipeModel
import warnings
warnings.filterwarnings('ignore')

data = read_and_transform_data('data/sum_12_fixed.jsonlines', homonym=True)

udpipe_model = UDPipeModel("data/20180506.uk.mova-institute.udpipe")
word_sense_detector = WordSenseDetector(
    pretrained_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    udpipe_model=udpipe_model,
    evaluation_dataset=data.head(100),
    prediction_strategy="max_sim_across_all_examples"
)
evaluation_dataset_pd = word_sense_detector.run()
# evaluation_dataset_pd = pd.read_csv('data/dummy_prediction.csv')

results_reports(evaluation_dataset_pd, udpipe_model)
