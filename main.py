"""
電信客戶流失分析 - 商業導向版本
基於Jerry建議的ROI驅動分析 + Expected Value (EV) 計算

核心理念：將數據分析轉換為可執行的賺錢行動
新增功能：Expected Value計算提供更強說服力
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class TelcoROIAnalyzer:
    """
    電信客戶流失ROI分析器 + Expected Value分析
    
    核心功能：
    1. 計算客戶生命週期價值(CLV)
    2. 識別高價值客戶
    3. 預測流失風險
    4. 計算挽回ROI
    5. 計算Expected Value (EV) - 增強說服力
    6. 生成可執行的行動方案
    """
    
    def __init__(self, data_path):
        """初始化分析器"""
        self.data_path = data_path
        self.df = None
        self.roi_framework = {
            'profit_margin': 0.6,  # 毛利率60%
            'acquisition_cost': 200,  # 獲客成本$200
            'retention_strategies': {
                'high_value': {'cost_range': (200, 600), 'success_rate': 0.5},
                'mid_value': {'cost_range': (100, 300), 'success_rate': 0.325},
                'low_value': {'cost_range': (20, 100), 'success_rate': 0.175}
            },
            # Expected Value 計算參數
            'ev_parameters': {
                'market_uncertainty': 0.15,  # 市場不確定性15%
                'execution_risk': 0.1,       # 執行風險10%
                'competitive_pressure': 0.05  # 競爭壓力5%
            }
        }
        
    def load_and_prepare_data(self):
        """載入並準備資料"""
        print("🔄 載入客戶資料...")
        self.df = pd.read_csv(self.data_path)
        
        # 基本資料清理
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
        self.df = self.df.dropna(subset=['TotalCharges'])
        
        # 轉換分類變數
        self.df['Churn'] = (self.df['Churn'] == 'Yes').astype(int)
        
        print(f"✅ 資料載入完成：{len(self.df)} 個客戶")
        return self.df
    
    def calculate_clv(self):
        """計算客戶生命週期價值(CLV)"""
        print("💰 計算客戶生命週期價值...")
        
        # 計算平均留存月數（基於tenure）
        self.df['EstimatedLifetime'] = np.where(
            self.df['Churn'] == 1,
            self.df['tenure'],  # 已流失客戶的實際留存時間
            self.df['tenure'] * 2  # 未流失客戶的預期留存時間（保守估計）
        )
        
        # 計算CLV
        self.df['CLV'] = (
            self.df['MonthlyCharges'] * 
            self.df['EstimatedLifetime'] * 
            self.roi_framework['profit_margin']
        )
        
        print(f"✅ CLV計算完成，平均CLV: ${self.df['CLV'].mean():.2f}")
        return self.df
    
    def segment_customers(self):
        """客戶價值分群"""
        print("🎯 進行客戶價值分群...")
        
        # 定義分群標準
        def categorize_customer(row):
            monthly = row['MonthlyCharges']
            clv = row['CLV']
            
            if monthly > 100 and clv > 2000:
                return 'High-Value'
            elif monthly >= 50 and clv >= 1000:
                return 'Mid-Value'
            else:
                return 'Low-Value'
        
        self.df['CustomerSegment'] = self.df.apply(categorize_customer, axis=1)
        
        # 分群統計
        segment_stats = self.df.groupby('CustomerSegment').agg({
            'customerID': 'count',
            'MonthlyCharges': 'mean',
            'CLV': 'mean',
            'Churn': 'mean'
        }).round(2)
        
        print("📊 客戶分群結果：")
        print(segment_stats)
        
        return segment_stats
    
    def predict_churn_risk(self):
        """預測流失風險"""
        print("🔮 建立流失預測模型...")
        
        try:
            # 準備特徵
            feature_columns = [
                'tenure', 'MonthlyCharges', 'TotalCharges',
                'Contract', 'PaymentMethod', 'InternetService'
            ]
            
            # 編碼分類變數
            df_encoded = pd.get_dummies(self.df[feature_columns])
            
            # 確保沒有無限值或缺失值
            df_encoded = df_encoded.fillna(0)
            df_encoded = df_encoded.replace([np.inf, -np.inf], 0)
            
            # 訓練模型
            X = df_encoded
            y = self.df['Churn']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # 使用較小的參數以提高速度
            self.churn_model = RandomForestClassifier(
                n_estimators=50, 
                max_depth=10,
                random_state=42,
                n_jobs=-1  # 使用所有CPU核心
            )
            self.churn_model.fit(X_train, y_train)
            
            # 預測流失概率
            self.df['ChurnProbability'] = self.churn_model.predict_proba(X)[:, 1]
            
            # 模型評估
            accuracy = self.churn_model.score(X_test, y_test)
            print(f"📈 模型準確率：{accuracy:.3f}")
            
            return self.churn_model
            
        except Exception as e:
            print(f"⚠️ 模型訓練錯誤: {e}")
            # 使用簡單的啟發式方法作為備選
            print("🔄 使用簡化預測方法...")
            
            # 基於tenure和contract的簡單流失概率
            self.df['ChurnProbability'] = np.where(
                (self.df['tenure'] < 12) & (self.df['Contract'] == 'Month-to-month'),
                0.8,  # 高風險
                np.where(
                    self.df['tenure'] < 24,
                    0.4,  # 中風險
                    0.1   # 低風險
                )
            )
            
            print("✅ 簡化預測完成")
            return None
    
    def calculate_retention_roi(self):
        """計算挽回ROI + Expected Value"""
        print("💡 計算挽回投資回報率和期望值...")
        
        def calculate_roi_and_ev(row):
            segment = row['CustomerSegment']
            clv = row['CLV']
            churn_prob = row['ChurnProbability']
            
            # 流失成本
            churn_cost = clv + self.roi_framework['acquisition_cost']
            
            # 挽回策略成本
            strategy = self.roi_framework['retention_strategies'][segment.lower().replace('-', '_')]
            retention_cost = np.mean(strategy['cost_range'])
            success_rate = strategy['success_rate']
            
            # ROI計算
            expected_return = success_rate * churn_cost
            roi = (expected_return - retention_cost) / retention_cost * 100
            
            # Expected Value (EV) 計算 - 修正版本
            # EV = (成功概率 × 成功收益) - 投資成本
            success_probability = success_rate
            
            # 考慮風險調整因子
            ev_params = self.roi_framework['ev_parameters']
            risk_adjustment = (1 - ev_params['market_uncertainty'] - 
                             ev_params['execution_risk'] - 
                             ev_params['competitive_pressure'])
            
            # 成功情境收益：保留客戶價值（風險調整後）
            success_value = clv * risk_adjustment
            
            # 期望收益
            expected_value = success_probability * success_value
            
            # 淨期望值（扣除投資成本）
            net_expected_value = expected_value - retention_cost
            
            # EV回報率
            ev_return_rate = (expected_value - retention_cost) / retention_cost * 100 if retention_cost > 0 else 0
            
            # 風險調整後的信心指數
            confidence_score = success_rate * risk_adjustment * 100
            
            # 調整推薦邏輯 - 降低門檻以獲得更實際的結果
            recommend_action = 'Execute' if (roi > 25 and net_expected_value > -50) else 'Skip'
            
            return {
                'ChurnCost': churn_cost,
                'RetentionCost': retention_cost,
                'ExpectedReturn': expected_return,
                'ROI': roi,
                'ExpectedValue': expected_value,
                'NetExpectedValue': net_expected_value,
                'EVReturnRate': ev_return_rate,
                'SuccessProbability': success_probability,
                'ConfidenceScore': confidence_score,
                'RecommendAction': recommend_action
            }
        
        # 只對高風險客戶計算ROI和EV
        high_risk_customers = self.df[self.df['ChurnProbability'] > 0.5].copy()
        
        roi_ev_results = high_risk_customers.apply(calculate_roi_and_ev, axis=1, result_type='expand')
        high_risk_customers = pd.concat([high_risk_customers, roi_ev_results], axis=1)
        
        self.high_risk_with_roi = high_risk_customers
        
        print(f"✅ 識別出 {len(high_risk_customers)} 個高風險客戶")
        print(f"🎯 建議執行挽回的客戶數: {len(high_risk_customers[high_risk_customers['RecommendAction'] == 'Execute'])}")
        print(f"💎 平均期望值: ${high_risk_customers['ExpectedValue'].mean():.2f}")
        print(f"💰 平均淨期望值: ${high_risk_customers['NetExpectedValue'].mean():.2f}")
        print(f"📊 平均信心指數: {high_risk_customers['ConfidenceScore'].mean():.1f}%")
        
        return high_risk_customers
    
    def generate_action_plan(self):
        """生成可執行的行動方案（包含EV分析）"""
        print("🚀 生成行動方案（包含期望值分析）...")
        
        # 按NetExpectedValue排序，優先處理高EV客戶
        actionable_customers = self.high_risk_with_roi[
            self.high_risk_with_roi['RecommendAction'] == 'Execute'
        ].sort_values('NetExpectedValue', ascending=False)
        
        # 按客戶分群生成具體行動
        action_plan = []
        
        for segment in ['High-Value', 'Mid-Value', 'Low-Value']:
            segment_customers = actionable_customers[
                actionable_customers['CustomerSegment'] == segment
            ]
            
            if len(segment_customers) > 0:
                avg_roi = segment_customers['ROI'].mean()
                avg_ev = segment_customers['ExpectedValue'].mean()
                avg_net_ev = segment_customers['NetExpectedValue'].mean()
                avg_ev_return = segment_customers['EVReturnRate'].mean()
                avg_confidence = segment_customers['ConfidenceScore'].mean()
                total_investment = segment_customers['RetentionCost'].sum()
                expected_return = segment_customers['ExpectedReturn'].sum()
                total_ev = segment_customers['ExpectedValue'].sum()
                total_net_ev = segment_customers['NetExpectedValue'].sum()
                
                action_plan.append({
                    'Segment': segment,
                    'CustomerCount': len(segment_customers),
                    'TotalInvestment': total_investment,
                    'ExpectedReturn': expected_return,
                    'AverageROI': avg_roi,
                    'TotalExpectedValue': total_ev,
                    'TotalNetExpectedValue': total_net_ev,
                    'AverageEV': avg_ev,
                    'AverageNetEV': avg_net_ev,
                    'AVGEVReturnRate': avg_ev_return,
                    'ConfidenceScore': avg_confidence,
                    'NetProfit': expected_return - total_investment
                })
        
        if len(action_plan) > 0:
            self.action_plan = pd.DataFrame(action_plan)
            
            print("📋 行動方案摘要（含期望值分析）：")
            for _, row in self.action_plan.iterrows():
                print(f"\n{row['Segment']} 客戶:")
                print(f"  客戶數量: {row['CustomerCount']}")
                print(f"  投資金額: ${row['TotalInvestment']:,.2f}")
                print(f"  期望回報: ${row['ExpectedReturn']:,.2f}")
                print(f"  總期望值: ${row['TotalExpectedValue']:,.2f}")
                print(f"  淨期望值: ${row['TotalNetExpectedValue']:,.2f}")
                print(f"  平均ROI: {row['AverageROI']:.1f}%")
                print(f"  平均EV回報率: {row['AVGEVReturnRate']:.1f}%")
                print(f"  信心指數: {row['ConfidenceScore']:.1f}%")
        else:
            # 創建空的DataFrame以避免錯誤
            self.action_plan = pd.DataFrame(columns=[
                'Segment', 'CustomerCount', 'TotalInvestment', 'ExpectedReturn',
                'AverageROI', 'TotalExpectedValue', 'TotalNetExpectedValue',
                'AverageEV', 'AverageNetEV', 'AVGEVReturnRate', 'ConfidenceScore', 'NetProfit'
            ])
            print("⚠️ 沒有符合執行條件的客戶")
        
        return self.action_plan
    
    def create_executive_summary(self):
        """創建高管摘要報告（包含EV分析）"""
        print("📊 生成高管摘要報告（包含期望值分析）...")
        
        # 計算關鍵指標
        total_customers = len(self.df)
        high_risk_customers = len(self.high_risk_with_roi)
        actionable_customers = len(self.high_risk_with_roi[
            self.high_risk_with_roi['RecommendAction'] == 'Execute'
        ])
        
        if len(self.action_plan) > 0:
            total_investment = self.action_plan['TotalInvestment'].sum()
            total_expected_return = self.action_plan['ExpectedReturn'].sum()
            total_expected_value = self.action_plan['TotalExpectedValue'].sum()
            total_net_expected_value = self.action_plan['TotalNetExpectedValue'].sum()
            total_net_profit = self.action_plan['NetProfit'].sum()
            overall_roi = (total_expected_return - total_investment) / total_investment * 100 if total_investment > 0 else 0
            overall_ev_return = total_net_expected_value / total_investment * 100 if total_investment > 0 else 0
            avg_confidence = self.action_plan['ConfidenceScore'].mean()
        else:
            total_investment = 0
            total_expected_return = 0
            total_expected_value = 0
            total_net_expected_value = 0
            total_net_profit = 0
            overall_roi = 0
            overall_ev_return = 0
            avg_confidence = 0
        
        summary = {
            'TotalCustomers': total_customers,
            'HighRiskCustomers': high_risk_customers,
            'ActionableCustomers': actionable_customers,
            'TotalInvestment': total_investment,
            'ExpectedReturn': total_expected_return,
            'TotalExpectedValue': total_expected_value,
            'TotalNetExpectedValue': total_net_expected_value,
            'NetProfit': total_net_profit,
            'OverallROI': overall_roi,
            'OverallEVReturn': overall_ev_return,
            'AverageConfidence': avg_confidence,
            'RevenueProtectionRate': actionable_customers / high_risk_customers * 100 if high_risk_customers > 0 else 0
        }
        
        print("🎯 專案ROI + EV 摘要：")
        print(f"總投資: ${total_investment:,.2f}")
        print(f"預期回報: ${total_expected_return:,.2f}")
        print(f"總期望值: ${total_expected_value:,.2f}")
        print(f"淨期望值: ${total_net_expected_value:,.2f}")
        print(f"淨利潤: ${total_net_profit:,.2f}")
        print(f"整體ROI: {overall_roi:.1f}%")
        print(f"整體EV回報率: {overall_ev_return:.1f}%")
        print(f"平均信心指數: {avg_confidence:.1f}%")
        print(f"收入保護率: {summary['RevenueProtectionRate']:.1f}%")
        
        return summary
    
    def visualize_roi_analysis(self):
        """視覺化ROI分析結果"""
        print("📈 生成ROI分析圖表...")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. 客戶分群與流失率
            segment_churn = self.df.groupby('CustomerSegment')['Churn'].mean()
            axes[0, 0].bar(segment_churn.index, segment_churn.values, color=['red', 'orange', 'green'])
            axes[0, 0].set_title('各客戶分群流失率')
            axes[0, 0].set_ylabel('流失率')
            
            # 2. ROI分布
            if len(self.high_risk_with_roi) > 0:
                axes[0, 1].hist(self.high_risk_with_roi['ROI'], bins=20, alpha=0.7, color='blue')
                axes[0, 1].axvline(50, color='red', linestyle='--', label='ROI門檻(50%)')
                axes[0, 1].set_title('挽回ROI分布')
                axes[0, 1].set_xlabel('ROI (%)')
                axes[0, 1].legend()
            else:
                axes[0, 1].text(0.5, 0.5, 'No high-risk customers found', 
                               horizontalalignment='center', verticalalignment='center',
                               transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('挽回ROI分布')
            
            # 3. 行動方案投資回報
            if len(self.action_plan) > 0:
                x_pos = range(len(self.action_plan))
                axes[1, 0].bar(x_pos, self.action_plan['TotalInvestment'], 
                              alpha=0.7, label='投資成本', color='red')
                axes[1, 0].bar(x_pos, self.action_plan['ExpectedReturn'], 
                              alpha=0.7, label='預期回報', color='green')
                axes[1, 0].set_title('各分群投資回報比較')
                axes[1, 0].set_xticks(x_pos)
                axes[1, 0].set_xticklabels(self.action_plan['Segment'])
                axes[1, 0].legend()
            else:
                axes[1, 0].text(0.5, 0.5, 'No actionable customers found', 
                               horizontalalignment='center', verticalalignment='center',
                               transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('各分群投資回報比較')
            
            # 4. 客戶價值vs流失風險 (sample for performance)
            sample_size = min(1000, len(self.df))
            df_sample = self.df.sample(n=sample_size, random_state=42)
            
            scatter_colors = df_sample['CustomerSegment'].map({
                'High-Value': 'red', 'Mid-Value': 'orange', 'Low-Value': 'green'
            })
            axes[1, 1].scatter(df_sample['CLV'], df_sample['ChurnProbability'], 
                              c=scatter_colors, alpha=0.6, s=20)
            axes[1, 1].set_title(f'客戶價值 vs 流失風險 (樣本: {sample_size})')
            axes[1, 1].set_xlabel('客戶生命週期價值 (CLV)')
            axes[1, 1].set_ylabel('流失概率')
            
            plt.tight_layout()
            plt.savefig('roi_analysis_dashboard.png', dpi=150, bbox_inches='tight')
            plt.close()  # Close the figure to free memory
            
            print("✅ 圖表已保存為 'roi_analysis_dashboard.png'")
            
        except Exception as e:
            print(f"⚠️ 視覺化過程中發生錯誤: {e}")
            print("✅ 繼續執行其他分析...")
    
    def run_complete_analysis(self):
        """執行完整的ROI分析流程"""
        print("🚀 開始執行完整ROI分析...")
        print("=" * 50)
        
        # 1. 載入資料
        self.load_and_prepare_data()
        
        # 2. 計算CLV
        self.calculate_clv()
        
        # 3. 客戶分群
        self.segment_customers()
        
        # 4. 預測流失風險
        self.predict_churn_risk()
        
        # 5. 計算ROI
        self.calculate_retention_roi()
        
        # 6. 生成行動方案
        self.generate_action_plan()
        
        # 7. 創建高管摘要
        self.create_executive_summary()
        
        # 8. 視覺化結果
        self.visualize_roi_analysis()
        
        print("=" * 50)
        print("✅ 完整ROI分析完成！")
        print("🎯 核心成果：數據已轉換為可執行的賺錢行動")
        print("💰 重點：專注高價值客戶，確保投資回報率")

def main():
    """主函數"""
    # 初始化分析器
    analyzer = TelcoROIAnalyzer('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # 執行完整分析
    analyzer.run_complete_analysis()
    
    print("\n🎉 分析完成！")
    print("📋 下一步：檢視行動方案，開始執行高ROI的客戶挽回策略")

if __name__ == "__main__":
    main() 