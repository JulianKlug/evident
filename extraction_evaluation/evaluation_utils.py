import pandas as pd
from thefuzz import process

from similarity_evaluation.similarity_models import SentenceTransformerSimilarityModel, SimilarityModel


def score_recommendation(recommendation_text:str, recommendation_grade:str, recommendation_level:str, recommendation_gt_df:pd.DataFrame,
                        semantic_model:SimilarityModel, semantic_threshold:float=0.6,
                         interactive:bool=True, verbose:bool=True) -> tuple:
    """
    This function takes a recommendation text, grade, and level, and returns the scores of the recommendation based on the ground truth dataframe.

    if recommendation text is not in the ground truth dataframe, it returns (-1, -1, -1, -1)
    if recommendation text is in the ground truth dataframe, TRUE/FALSE for grade and level based on if the recommendation grade and level are the same as the ground truth dataframe, as well as the matched recommendation text and the score of the match.

    Parameters
    ----------
    recommendation_text : str
        The recommendation text to be scored.
    recommendation_grade : str
        The recommendation grade to be scored.
    recommendation_level : str
        The recommendation level to be scored.
    recommendation_gt_df : pd.DataFrame
        The ground truth dataframe containing the recommendation text, grade, and level.
    semantic_model : SimilarityModel
        The semantic model to be used for scoring the recommendation text.
    semantic_threshold : float, optional
        The threshold for the semantic similarity score to consider a match. Default is 0.6
    interactive : bool, optional
        If True, the function will prompt the user to choose from the best matches if no exact match is found. Default is True.
    verbose : bool
        If True, print the fuzzy match found for the recommendation text.


    Returns
    -------
    tuple
        A tuple containing the grade evaluation, level evaluation, matched recommendation text, score of the match, as well as manual validation
        (grade_eval, level_eval, recommendation_text, score, manual_validation)

    """

    # Check if exact match for recommendation text in the ground truth dataframe
    if recommendation_text in recommendation_gt_df.recommendation.values:
        match_row = recommendation_gt_df[recommendation_gt_df.recommendation.str.lower() == recommendation_text.lower()]
        grade_eval = recommendation_grade.lower() == match_row['class'].values[0].lower()
        level_eval = recommendation_level.lower() == match_row.LOE.values[0].lower()
        return (grade_eval, level_eval, recommendation_text, 100, False)

    # Check if fuzzy match for recommendation text in the ground truth dataframe
    matches = process.extractBests(recommendation_text.lower(), recommendation_gt_df.recommendation.str.lower().values,
                                   limit=10)

    best_match = matches[0]
    if best_match[1] >= 95:
        match_row = recommendation_gt_df[recommendation_gt_df.recommendation.str.lower() == best_match[0].lower()]
        grade_eval = recommendation_grade.lower() == match_row['class'].values[0].lower()
        level_eval = recommendation_level.lower() == match_row.LOE.values[0].lower()

        if verbose:
            print(f'Fuzzy match found for "{recommendation_text}" with score {best_match[1]}: "{best_match[0]}"')

        return (grade_eval, level_eval, best_match[0], best_match[1], False)

    else:
        # if not complete match is found, use get_similarity_score to find the best match
        similarities = []
        for i, row in recommendation_gt_df.iterrows():
            score = semantic_model.compute_similarity(recommendation_text, row['recommendation'])
            similarities.append((row['recommendation'], score))

        # Find the best semantic match
        best_semantic_match = max(similarities, key=lambda x: x[1])
        if best_semantic_match[1] >= semantic_threshold:
            match_row = recommendation_gt_df[recommendation_gt_df.recommendation == best_semantic_match[0]]
            grade_eval = recommendation_grade.lower() == match_row['class'].values[0].lower()
            level_eval = recommendation_level.lower() == match_row.LOE.values[0].lower()
            
            if verbose:
                print(f'Semantic match found for "{recommendation_text}" with score {best_semantic_match[1]:.3f}: "{best_semantic_match[0]}"')
            
            return (grade_eval, level_eval, best_semantic_match[0], best_semantic_match[1] * 100, False)
        
        if not interactive:
            if verbose:
                print(f'No match found for "{recommendation_text}"')
            return (-1, -1, -1, -1, False)
        else:
            # give choice of best matches to the user
            print('No exact match found, please choose from the following options:')
            print(f'- "{recommendation_text} -"')
            for i, match in enumerate(matches):
                print(f'{i}: {match[0]} - {match[1]}')
            # add none option
            print(f'{len(matches)}: None')

            # get user input
            nl = '\n'
            user_choice = input(f'No exact match found, please choose from the following options:'
                                f'\n- "{recommendation_text} -"'
                                f'\n{nl.join([f"{i}: {match[0]} - {match[1]}" for i, match in enumerate(matches)])}'
                                f'\n{len(matches)}: None'
                                f'\nPlease enter the number of your choice: ')

            # check if user input is valid
            if user_choice.isdigit() and int(user_choice) < len(matches):
                match_row = recommendation_gt_df[recommendation_gt_df.recommendation.str.lower() == matches[int(user_choice)][0].lower()]
                grade_eval = recommendation_grade.lower() == match_row['class'].values[0].lower()
                level_eval = recommendation_level.lower() == match_row.LOE.values[0].lower()
                return (grade_eval, level_eval, matches[int(user_choice)][0], matches[int(user_choice)][1], True)

            # no match found
            elif user_choice.isdigit() and int(user_choice) == len(matches):
                return (-1, -1, -1, -1, True)
            else:
                print('Invalid choice, no match retained')
                return (-1, -1, -1, -1, False)


def evaluate_guideline_extraction(df:pd.DataFrame, gt_df:pd.DataFrame, interactive:bool=True, verbose:bool=True) -> tuple:
    """
    Evaluate an extracted guideline recommendation dataframe against a ground truth dataframe.
    The function iterates through the extracted dataframe, scoring each recommendation based on its text, grade, and level.
    Accuracy for grade and level is calculated as the percentage of correct matches.

    Parameters
    ----------
    :param df:
    :param gt_df:
    :param interactive:
    :param verbose:

    Returns
    -------
    :return: tuple
        A tuple containing the accuracy of recommendations, accuracy of grades, accuracy of levels, number of missing recommendations,
        a dataframe of all matches, and a dataframe of missing recommendations.
        (accuracy_recommendation, accuracy_grade, accuracy_level, n_missing_recommendations, all_matches_df, missing_recommendations_df)
    """
    # Initialize variables
    n_recommendations = len(df)
    n_correct_recommendations = 0
    n_correct_grades = 0
    n_correct_levels = 0
    all_matches = []
    semantic_model = SentenceTransformerSimilarityModel("neuml/pubmedbert-base-embeddings")

    # Iterate through the extracted dataframe
    for i, row in df.iterrows():
        recommendation_text = row['recommendation']
        recommendation_grade = row['class']
        recommendation_level = row['LOE']

        # Score the recommendation
        grade_eval, level_eval, matched_text, score, manual_validation = score_recommendation(recommendation_text,
                                                                                             recommendation_grade,
                                                                                             recommendation_level,
                                                                                             gt_df,
                                                                                             semantic_model=semantic_model,
                                                                                             interactive=interactive,
                                                                                             verbose=verbose)

        # Update counts based on evaluation
        if grade_eval == True:
            n_correct_grades += 1
        if level_eval == True:
            n_correct_levels += 1
        if matched_text != -1:
            n_correct_recommendations += 1

        # Append match to all matches list
        all_matches.append((recommendation_text, recommendation_grade, recommendation_level, matched_text, grade_eval, level_eval, score, manual_validation))

    # Calculate accuracies
    accuracy_recommendation = n_correct_recommendations / n_recommendations * 100

    # avoid division by zero
    if n_correct_recommendations == 0:
        accuracy_grade = 0
        accuracy_level = 0
    else:
        accuracy_grade = n_correct_grades / n_correct_recommendations * 100
        accuracy_level = n_correct_levels / n_correct_recommendations * 100

    all_matches_df = pd.DataFrame(all_matches, columns=['recommendation_text', 'recommendation_grade', 'recommendation_level', 'matched_text', 'grade_eval', 'level_eval', 'match_score', 'manual_validation'])

    # missing recommendations (ie in ground truth but not in extracted)
    missing_recommendations_df = gt_df[~gt_df.recommendation.isin(df.recommendation.values)]
    n_missing_recommendations = len(missing_recommendations_df)

    return (accuracy_recommendation, accuracy_grade, accuracy_level, n_missing_recommendations, all_matches_df, missing_recommendations_df)


if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser(description='Evaluate guideline extraction')
    parser.add_argument('-p', '--path', type=str, help='Path to the extracted dataframe')
    parser.add_argument('-g', '--gt_path', type=str, help='Path to the ground truth dataframe')
    parser.add_argument('-ni', '--not_interactive', action='store_true', help='Run in non-interactive mode')
    parser.add_argument('-v', '--verbose', action='store_true', help='Run in verbose mode')

    args = parser.parse_args()

    interactive = not args.not_interactive
    df = pd.read_csv(args.path)
    gt_df = pd.read_csv(args.gt_path)
    accuracy_recommendation, accuracy_grade, accuracy_level, n_missing_recommendations, all_matches_df, missing_recommendations_df = evaluate_guideline_extraction(df, gt_df, interactive=interactive, verbose=args.verbose)
    print(f'Accuracy of recommendations: {accuracy_recommendation:.2f}%')
    print(f'Accuracy of grades: {accuracy_grade:.2f}%')
    print(f'Accuracy of levels: {accuracy_level:.2f}%')
    print(f'Number of missing recommendations: {n_missing_recommendations}')

    # save all matches and missing recommendations to csv
    output_dir = os.path.dirname(args.path)
    all_matches_df.to_csv(os.path.join(output_dir, 'all_matches.csv'), index=False)
    missing_recommendations_df.to_csv(os.path.join(output_dir, 'missing_recommendations.csv'), index=False)
