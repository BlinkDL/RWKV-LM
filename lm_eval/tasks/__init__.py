from pprint import pprint
from typing import List, Union

import sacrebleu
import lm_eval.base

from . import superglue
from . import glue
from . import arc
from . import coqa
from . import race
from . import webqs
from . import anli
from . import wsc273
from . import winogrande
from . import quac
from . import hellaswag
from . import swag
from . import openbookqa
from . import squad
from . import naturalqs
from . import sat
from . import arithmetic
from . import lambada
from . import piqa
from . import prost
from . import mc_taco
from . import triviaqa
from . import pubmedqa
from . import sciq
from . import qasper
from . import qa4mre
from . import translation
from . import headqa
from . import mathqa
from . import hendrycks_ethics
from . import drop
from . import unscramble
from . import logiqa
from . import hendrycks_test
from . import hendrycks_math
from . import cbt
from . import lambada_cloze
from . import pile
from . import wikitext
from . import lambada_multilingual
from . import mutual
from . import truthfulqa
from . import blimp
from . import asdiv
from . import gsm8k
from . import storycloze

########################################
# Translation tasks
########################################

# 6 total
gpt3_translation_benchmarks = {
    "wmt14": ["en-fr", "fr-en"],  # French
    "wmt16": ["en-ro", "ro-en", "de-en", "en-de"],  # German, Romanian
}


# 28 total
selected_translation_benchmarks = {
    **gpt3_translation_benchmarks,
    "wmt20": sacrebleu.get_langpairs_for_testset("wmt20"),
    "iwslt17": ["en-ar", "ar-en"],  # Arabic
}

# 319 total
all_translation_benchmarks = {
    ts: sacrebleu.get_langpairs_for_testset(ts)
    for ts in sacrebleu.get_available_testsets()
}


########################################
# All tasks
########################################


TASK_REGISTRY = {
    # GLUE
    "cola": glue.CoLA,
    "mnli": glue.MNLI,
    "mnli_mismatched": glue.MNLIMismatched,
    "mrpc": glue.MRPC,
    "rte": glue.RTE,
    "qnli": glue.QNLI,
    "qqp": glue.QQP,
    # "stsb": glue.STSB, # not implemented yet
    "sst": glue.SST,
    "wnli": glue.WNLI,
    # SuperGLUE
    "boolq": superglue.BoolQ,
    "cb": superglue.CommitmentBank,
    "copa": superglue.Copa,
    "multirc": superglue.MultiRC,
    "record": superglue.ReCoRD,
    "wic": superglue.WordsInContext,
    "wsc": superglue.SGWinogradSchemaChallenge,
    # Order by benchmark/genre?
    "coqa": coqa.CoQA,
    "drop": drop.DROP,
    "lambada_openai": lambada.LambadaOpenAI,
    "lambada_standard": lambada.LambadaStandard,
    "lambada_openai_cloze": lambada_cloze.LambadaOpenAICloze,
    "lambada_standard_cloze": lambada_cloze.LambadaStandardCloze,
    # multilingual lambada
    **lambada_multilingual.construct_tasks(),
    "wikitext": wikitext.WikiText,
    # "cbt-cn": cbt.CBTCN, # disabled pending context length fix
    # "cbt-ne": cbt.CBTNE, # disabled pending context length fix
    "piqa": piqa.PiQA,
    "prost": prost.PROST,
    "mc_taco": mc_taco.MCTACO,
    # Science related
    "pubmedqa": pubmedqa.Pubmed_QA,
    "sciq": sciq.SciQ,
    "qasper": qasper.QASPER,
    "qa4mre_2011": qa4mre.QA4MRE_2011,
    "qa4mre_2012": qa4mre.QA4MRE_2012,
    "qa4mre_2013": qa4mre.QA4MRE_2013,
    "triviaqa": triviaqa.TriviaQA,
    "arc_easy": arc.ARCEasy,
    "arc_challenge": arc.ARCChallenge,
    # "quac": quac.QuAC, # not implemented yet
    "logiqa": logiqa.LogiQA,
    "hellaswag": hellaswag.HellaSwag,
    "swag": swag.SWAG,
    "openbookqa": openbookqa.OpenBookQA,
    "squad2": squad.SQuAD2,
    "race": race.RACE,
    # "naturalqs": naturalqs.NaturalQs, # not implemented yet
    "headqa": headqa.HeadQAEsDeprecated,  # for backwards compat - headqa used to default to es
    "headqa_es": headqa.HeadQAEs,
    "headqa_en": headqa.HeadQAEn,
    "mathqa": mathqa.MathQA,
    "webqs": webqs.WebQs,
    "wsc273": wsc273.WinogradSchemaChallenge273,
    "winogrande": winogrande.Winogrande,
    "anli_r1": anli.ANLIRound1,
    "anli_r2": anli.ANLIRound2,
    "anli_r3": anli.ANLIRound3,
    "ethics_cm": hendrycks_ethics.EthicsCM,
    "ethics_deontology": hendrycks_ethics.EthicsDeontology,
    "ethics_justice": hendrycks_ethics.EthicsJustice,
    "ethics_utilitarianism_original": hendrycks_ethics.EthicsUtilitarianismOriginal,
    "ethics_utilitarianism": hendrycks_ethics.EthicsUtilitarianism,
    "ethics_virtue": hendrycks_ethics.EthicsVirtue,
    "truthfulqa_mc": truthfulqa.TruthfulQAMultipleChoice,
    "truthfulqa_gen": truthfulqa.TruthfulQAGeneration,
    # dialogue
    "mutual": mutual.MuTual,
    "mutual_plus": mutual.MuTualPlus,
    # math
    "math_algebra": hendrycks_math.MathAlgebra,
    "math_counting_and_prob": hendrycks_math.MathCountingAndProbability,
    "math_geometry": hendrycks_math.MathGeometry,
    "math_intermediate_algebra": hendrycks_math.MathIntermediateAlgebra,
    "math_num_theory": hendrycks_math.MathNumberTheory,
    "math_prealgebra": hendrycks_math.MathPrealgebra,
    "math_precalc": hendrycks_math.MathPrecalculus,
    "math_asdiv": asdiv.Asdiv,
    "gsm8k": gsm8k.GradeSchoolMath8K,
    # arithmetic
    "arithmetic_2da": arithmetic.Arithmetic2DPlus,
    "arithmetic_2ds": arithmetic.Arithmetic2DMinus,
    "arithmetic_3da": arithmetic.Arithmetic3DPlus,
    "arithmetic_3ds": arithmetic.Arithmetic3DMinus,
    "arithmetic_4da": arithmetic.Arithmetic4DPlus,
    "arithmetic_4ds": arithmetic.Arithmetic4DMinus,
    "arithmetic_5da": arithmetic.Arithmetic5DPlus,
    "arithmetic_5ds": arithmetic.Arithmetic5DMinus,
    "arithmetic_2dm": arithmetic.Arithmetic2DMultiplication,
    "arithmetic_1dc": arithmetic.Arithmetic1DComposite,
    # TODO Perhaps make these groups of tasks
    #   e.g. anli, arithmetic, openai_translations, harness_translations
    # hendrycksTest (57 tasks)
    **hendrycks_test.create_all_tasks(),
    # e.g. wmt14-fr-en
    **translation.create_tasks_from_benchmarks(gpt3_translation_benchmarks),
    # chef's selection, mostly wmt20
    **translation.create_tasks_from_benchmarks(selected_translation_benchmarks),
    # Word Scrambling and Manipulation Tasks
    "anagrams1": unscramble.Anagrams1,
    "anagrams2": unscramble.Anagrams2,
    "cycle_letters": unscramble.CycleLetters,
    "random_insertion": unscramble.RandomInsertion,
    "reversed_words": unscramble.ReversedWords,
    # Pile
    "pile_arxiv": pile.PileArxiv,
    "pile_books3": pile.PileBooks3,
    "pile_bookcorpus2": pile.PileBookCorpus2,
    "pile_dm-mathematics": pile.PileDmMathematics,
    "pile_enron": pile.PileEnron,
    "pile_europarl": pile.PileEuroparl,
    "pile_freelaw": pile.PileFreeLaw,
    "pile_github": pile.PileGithub,
    "pile_gutenberg": pile.PileGutenberg,
    "pile_hackernews": pile.PileHackernews,
    "pile_nih-exporter": pile.PileNIHExporter,
    "pile_opensubtitles": pile.PileOpenSubtitles,
    "pile_openwebtext2": pile.PileOpenWebText2,
    "pile_philpapers": pile.PilePhilPapers,
    "pile_pile-cc": pile.PilePileCc,
    "pile_pubmed-abstracts": pile.PilePubmedAbstracts,
    "pile_pubmed-central": pile.PilePubmedCentral,
    "pile_stackexchange": pile.PileStackExchange,
    "pile_uspto": pile.PileUspto,
    "pile_ubuntu-irc": pile.PileUbuntuIrc,
    "pile_wikipedia": pile.PileWikipedia,
    "pile_youtubesubtitles": pile.PileYoutubeSubtitles,
    # BLiMP
    "blimp_adjunct_island": blimp.BlimpAdjunctIsland,
    "blimp_anaphor_gender_agreement": blimp.BlimpAnaphorGenderAgreement,
    "blimp_anaphor_number_agreement": blimp.BlimpAnaphorNumberAgreement,
    "blimp_animate_subject_passive": blimp.BlimpAnimateSubjectPassive,
    "blimp_animate_subject_trans": blimp.BlimpAnimateSubjectTrans,
    "blimp_causative": blimp.BlimpCausative,
    "blimp_complex_NP_island": blimp.BlimpComplex_NPIsland,
    "blimp_coordinate_structure_constraint_complex_left_branch": blimp.BlimpCoordinateStructureConstraintComplexLeftBranch,
    "blimp_coordinate_structure_constraint_object_extraction": blimp.BlimpCoordinateStructureConstraintObjectExtraction,
    "blimp_determiner_noun_agreement_1": blimp.BlimpDeterminerNounAgreement_1,
    "blimp_determiner_noun_agreement_2": blimp.BlimpDeterminerNounAgreement_2,
    "blimp_determiner_noun_agreement_irregular_1": blimp.BlimpDeterminerNounAgreementIrregular_1,
    "blimp_determiner_noun_agreement_irregular_2": blimp.BlimpDeterminerNounAgreementIrregular_2,
    "blimp_determiner_noun_agreement_with_adj_2": blimp.BlimpDeterminerNounAgreementWithAdj_2,
    "blimp_determiner_noun_agreement_with_adj_irregular_1": blimp.BlimpDeterminerNounAgreementWithAdjIrregular_1,
    "blimp_determiner_noun_agreement_with_adj_irregular_2": blimp.BlimpDeterminerNounAgreementWithAdjIrregular_2,
    "blimp_determiner_noun_agreement_with_adjective_1": blimp.BlimpDeterminerNounAgreementWithAdjective_1,
    "blimp_distractor_agreement_relational_noun": blimp.BlimpDistractorAgreementRelationalNoun,
    "blimp_distractor_agreement_relative_clause": blimp.BlimpDistractorAgreementRelativeClause,
    "blimp_drop_argument": blimp.BlimpDropArgument,
    "blimp_ellipsis_n_bar_1": blimp.BlimpEllipsisNBar_1,
    "blimp_ellipsis_n_bar_2": blimp.BlimpEllipsisNBar_2,
    "blimp_existential_there_object_raising": blimp.BlimpExistentialThereObjectRaising,
    "blimp_existential_there_quantifiers_1": blimp.BlimpExistentialThereQuantifiers_1,
    "blimp_existential_there_quantifiers_2": blimp.BlimpExistentialThereQuantifiers_2,
    "blimp_existential_there_subject_raising": blimp.BlimpExistentialThereSubjectRaising,
    "blimp_expletive_it_object_raising": blimp.BlimpExpletiveItObjectRaising,
    "blimp_inchoative": blimp.BlimpInchoative,
    "blimp_intransitive": blimp.BlimpIntransitive,
    "blimp_irregular_past_participle_adjectives": blimp.BlimpIrregularPastParticipleAdjectives,
    "blimp_irregular_past_participle_verbs": blimp.BlimpIrregularPastParticipleVerbs,
    "blimp_irregular_plural_subject_verb_agreement_1": blimp.BlimpIrregularPluralSubjectVerbAgreement_1,
    "blimp_irregular_plural_subject_verb_agreement_2": blimp.BlimpIrregularPluralSubjectVerbAgreement_2,
    "blimp_left_branch_island_echo_question": blimp.BlimpLeftBranchIslandEchoQuestion,
    "blimp_left_branch_island_simple_question": blimp.BlimpLeftBranchIslandSimpleQuestion,
    "blimp_matrix_question_npi_licensor_present": blimp.BlimpMatrixQuestionNpiLicensorPresent,
    "blimp_npi_present_1": blimp.BlimpNpiPresent_1,
    "blimp_npi_present_2": blimp.BlimpNpiPresent_2,
    "blimp_only_npi_licensor_present": blimp.BlimpOnlyNpiLicensorPresent,
    "blimp_only_npi_scope": blimp.BlimpOnlyNpiScope,
    "blimp_passive_1": blimp.BlimpPassive_1,
    "blimp_passive_2": blimp.BlimpPassive_2,
    "blimp_principle_A_c_command": blimp.BlimpPrinciple_ACCommand,
    "blimp_principle_A_case_1": blimp.BlimpPrinciple_ACase_1,
    "blimp_principle_A_case_2": blimp.BlimpPrinciple_ACase_2,
    "blimp_principle_A_domain_1": blimp.BlimpPrinciple_ADomain_1,
    "blimp_principle_A_domain_2": blimp.BlimpPrinciple_ADomain_2,
    "blimp_principle_A_domain_3": blimp.BlimpPrinciple_ADomain_3,
    "blimp_principle_A_reconstruction": blimp.BlimpPrinciple_AReconstruction,
    "blimp_regular_plural_subject_verb_agreement_1": blimp.BlimpRegularPluralSubjectVerbAgreement_1,
    "blimp_regular_plural_subject_verb_agreement_2": blimp.BlimpRegularPluralSubjectVerbAgreement_2,
    "blimp_sentential_negation_npi_licensor_present": blimp.BlimpSententialNegationNpiLicensorPresent,
    "blimp_sentential_negation_npi_scope": blimp.BlimpSententialNegationNpiScope,
    "blimp_sentential_subject_island": blimp.BlimpSententialSubjectIsland,
    "blimp_superlative_quantifiers_1": blimp.BlimpSuperlativeQuantifiers_1,
    "blimp_superlative_quantifiers_2": blimp.BlimpSuperlativeQuantifiers_2,
    "blimp_tough_vs_raising_1": blimp.BlimpToughVsRaising_1,
    "blimp_tough_vs_raising_2": blimp.BlimpToughVsRaising_2,
    "blimp_transitive": blimp.BlimpTransitive,
    "blimp_wh_island": blimp.BlimpWhIsland,
    "blimp_wh_questions_object_gap": blimp.BlimpWhQuestionsObjectGap,
    "blimp_wh_questions_subject_gap": blimp.BlimpWhQuestionsSubjectGap,
    "blimp_wh_questions_subject_gap_long_distance": blimp.BlimpWhQuestionsSubjectGapLongDistance,
    "blimp_wh_vs_that_no_gap": blimp.BlimpWhVsThatNoGap,
    "blimp_wh_vs_that_no_gap_long_distance": blimp.BlimpWhVsThatNoGapLongDistance,
    "blimp_wh_vs_that_with_gap": blimp.BlimpWhVsThatWithGap,
    "blimp_wh_vs_that_with_gap_long_distance": blimp.BlimpWhVsThatWithGapLongDistance,
    # Requires manual download of data.
    # "storycloze_2016": storycloze.StoryCloze2016,
    # "storycloze_2018": storycloze.StoryCloze2018,
    # "sat": sat.SATAnalogies,
}


ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name):
    try:
        return TASK_REGISTRY[task_name]
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")


def get_task_name_from_object(task_object):
    for name, class_ in TASK_REGISTRY.items():
        if class_ is task_object:
            return name

    # this gives a mechanism for non-registered tasks to have a custom name anyways when reporting
    return (
        task_object.EVAL_HARNESS_NAME
        if hasattr(task_object, "EVAL_HARNESS_NAME")
        else type(task_object).__name__
    )


def get_task_dict(task_name_list: List[Union[str, lm_eval.base.Task]]):
    task_name_dict = {
        task_name: get_task(task_name)()
        for task_name in task_name_list
        if isinstance(task_name, str)
    }
    task_name_from_object_dict = {
        get_task_name_from_object(task_object): task_object
        for task_object in task_name_list
        if not isinstance(task_object, str)
    }
    assert set(task_name_dict.keys()).isdisjoint(set(task_name_from_object_dict.keys()))
    return {**task_name_dict, **task_name_from_object_dict}
