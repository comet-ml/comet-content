import argparse
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--outfile')

    return parser.parse_args()


def main():
    args = get_args()

    df = pd.read_csv(args.data)

    """Some of these features were borrowed from work found in:
    https://github.com/neptune-ml/open-solution-home-credit/wiki/LightGBM-on-selected-features
    """
    features = pd.DataFrame()
    features[
        'employment_to_birth_ratio'] = df['DAYS_EMPLOYED'] / \
        df['DAYS_BIRTH']

    features[
        'credit_to_annuity_ratio'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']

    features['credit_to_goods_ratio'] = df['AMT_CREDIT'] / \
        df['AMT_GOODS_PRICE']

    features['credit_to_income_ratio'] = df['AMT_CREDIT'] / \
        df['AMT_INCOME_TOTAL']

    features['income_credit_percentage'] = df['AMT_INCOME_TOTAL'] / \
        df['AMT_CREDIT']

    features['income_per_child'] = df['AMT_INCOME_TOTAL'] / \
        (1 + df['CNT_CHILDREN'])

    features['income_per_person'] = df['AMT_INCOME_TOTAL'] / \
        df['CNT_FAM_MEMBERS']

    features['payment_rate'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

    features['amt_balance_to_income'] = df['CREDIT_BALANCE_AMT_BALANCE'] / \
        df['AMT_INCOME_TOTAL']

    features['no_inquiries_MON_to_birth'] = df['AMT_REQ_CREDIT_BUREAU_MON'] / \
        df['DAYS_BIRTH']

    features['no_inquiries_DAY_to_birth'] = df['AMT_REQ_CREDIT_BUREAU_DAY'] / \
        df['DAYS_BIRTH']

    features[
        'no_inquiries_WEEK_to_birth'] = df['AMT_REQ_CREDIT_BUREAU_WEEK'] / \
        df['DAYS_BIRTH']

    features['avg_external_source'] = (
        df['EXT_SOURCE_1'] + df['EXT_SOURCE_2'] + df['EXT_SOURCE_3']) / 3

    features['income_avg_external_source'] = df['AMT_INCOME_TOTAL'] / \
        features['avg_external_source']

    features['social_circle_obs_30_to_income'] = df['OBS_30_CNT_SOCIAL_CIRCLE'] / \
        df['AMT_INCOME_TOTAL']

    features['social_circle_def_30_to_income'] = df['DEF_30_CNT_SOCIAL_CIRCLE'] / \
        df['AMT_INCOME_TOTAL']

    features['social_circle_obs_60_to_income'] = df['OBS_60_CNT_SOCIAL_CIRCLE'] / \
        df['AMT_INCOME_TOTAL']

    features['social_circle_def_60_to_income'] = df['OBS_60_CNT_SOCIAL_CIRCLE'] / \
        df['AMT_INCOME_TOTAL']

    final = pd.concat([df, features], axis=1)
    final.to_csv(args.outfile, index=False)


if __name__ == '__main__':
    main()
