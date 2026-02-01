"""
Synthetic Data Generator for Sales Forecasting Platform
Generate realistic sales data for 500 stores across 40+ brands in 8 countries

Run this first to create all datasets!
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import random
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class SyntheticDataGenerator:
    """Generate synthetic sales forecasting data matching Chalhoub Group use case"""
    
    def __init__(self):
        # Configuration based on the presentation
        self.num_stores = 500
        self.num_brands = 45
        self.countries = ['UAE', 'Saudi Arabia', 'Kuwait', 'Qatar', 'Bahrain', 
                         'Oman', 'Egypt', 'Jordan']
        
        # Store tiers based on data richness
        self.data_rich_stores = 150  # 3+ years of data (30%)
        self.data_medium_stores = 250  # 1-3 years of data (50%)
        self.data_poor_stores = 100  # <1 year of data (20%)
        
        # Date range
        self.end_date = datetime.now()
        self.max_history_days = 1095  # 3 years
        
        print("="*70)
        print(" Sales Forecasting Platform - Synthetic Data Generator")
        print("="*70)
        print(f"Configuration:")
        print(f"  • Stores: {self.num_stores}")
        print(f"  • Brands: {self.num_brands}")
        print(f"  • Countries: {len(self.countries)}")
        print(f"  • Data Rich Stores: {self.data_rich_stores} (3+ years)")
        print(f"  • Data Medium Stores: {self.data_medium_stores} (1-3 years)")
        print(f"  • Data Poor Stores: {self.data_poor_stores} (<1 year)")
        print("="*70)
    
    def generate_store_metadata(self):
        """Generate store metadata with hierarchical structure"""
        print("\ Generating store metadata...")
        stores = []
        store_id = 1
        
        for _ in range(self.num_stores):
            # Determine data tier
            if store_id <= self.data_rich_stores:
                tier = 'data_rich'
                history_days = random.randint(1095, 1460)  # 3-4 years
            elif store_id <= self.data_rich_stores + self.data_medium_stores:
                tier = 'data_medium'
                history_days = random.randint(365, 1095)  # 1-3 years
            else:
                tier = 'data_poor'
                history_days = random.randint(180, 365)  # 6-12 months
            
            store = {
                'store_id': f'STORE_{store_id:03d}',
                'store_name': f'Store {store_id}',
                'brand_id': f'BRAND_{random.randint(1, self.num_brands):02d}',
                'country': random.choice(self.countries),
                'location_type': random.choice(['mall', 'street', 'airport']),
                'store_size_sqm': random.randint(50, 500),
                'tier': tier,
                'history_days': history_days,
                'opening_date': (self.end_date - timedelta(days=history_days)).strftime('%Y-%m-%d'),
                'is_ecommerce': False
            }
            stores.append(store)
            store_id += 1
        
        # Add e-commerce stores (one per country)
        print("  • Adding e-commerce stores (1 per country)...")
        for country in self.countries:
            store = {
                'store_id': f'ECOM_{country[:3].upper()}',
                'store_name': f'E-commerce {country}',
                'brand_id': 'BRAND_MULTI',
                'country': country,
                'location_type': 'online',
                'store_size_sqm': 0,
                'tier': 'data_rich',
                'history_days': 1095,
                'opening_date': (self.end_date - timedelta(days=1095)).strftime('%Y-%m-%d'),
                'is_ecommerce': True
            }
            stores.append(store)
        
        df = pd.DataFrame(stores)
        print(f"  Created {len(df)} stores")
        return df
    
    def generate_brand_metadata(self):
        """Generate brand metadata"""
        print("\n  Generating brand metadata...")
        brands = []
        
        brand_categories = ['Luxury Watches', 'Fashion', 'Cosmetics', 'Jewelry', 
                           'Handbags', 'Footwear', 'Eyewear', 'Fragrances']
        
        for i in range(1, self.num_brands + 1):
            brand = {
                'brand_id': f'BRAND_{i:02d}',
                'brand_name': f'Brand {i}',
                'category': random.choice(brand_categories),
                'price_tier': random.choice(['ultra_luxury', 'luxury', 'premium']),
                'avg_price_point': random.randint(100, 5000),
                'seasonality_strength': random.uniform(0.1, 0.5)
            }
            brands.append(brand)
        
        # Multi-brand for e-commerce
        brands.append({
            'brand_id': 'BRAND_MULTI',
            'brand_name': 'Multi-Brand Platform',
            'category': 'Multiple',
            'price_tier': 'mixed',
            'avg_price_point': 500,
            'seasonality_strength': 0.3
        })
        
        df = pd.DataFrame(brands)
        print(f"  ✓ Created {len(df)} brands")
        return df
    
    def generate_sales_data(self, stores_df, brands_df):
        """Generate daily sales data with realistic patterns"""
        print("\n💰 Generating sales data (this takes a moment)...")
        all_sales = []
        
        total_stores = len(stores_df)
        for idx, (_, store) in enumerate(stores_df.iterrows(), 1):
            if idx % 50 == 0:
                print(f"  • Processing store {idx}/{total_stores}...")
            
            # Get brand info
            brand = brands_df[brands_df['brand_id'] == store['brand_id']].iloc[0]
            
            # Generate dates for this store
            start_date = datetime.strptime(store['opening_date'], '%Y-%m-%d')
            dates = pd.date_range(start=start_date, end=self.end_date, freq='D')
            
            # Base sales level (influenced by store size and brand price)
            base_sales = (store['store_size_sqm'] * brand['avg_price_point'] * 
                         random.uniform(0.5, 1.5) / 100)
            
            if store['is_ecommerce']:
                base_sales *= 3  # E-commerce has higher volume
            
            for date in dates:
                # Trend component (slow growth)
                days_since_open = (date - start_date).days
                trend = 1 + (days_since_open / 3650) * random.uniform(0.1, 0.3)
                
                # Seasonality (annual + weekly)
                day_of_year = date.timetuple().tm_yday
                annual_seasonality = 1 + brand['seasonality_strength'] * np.sin(
                    2 * np.pi * day_of_year / 365
                )
                
                day_of_week = date.weekday()
                weekly_seasonality = {
                    0: 0.8, 1: 0.85, 2: 0.9, 3: 0.95,  # Mon-Thu
                    4: 1.2, 5: 1.3, 6: 1.1  # Fri-Sun (weekend boost)
                }[day_of_week]
                
                # Special events (Ramadan, Black Friday, National Days)
                special_multiplier = 1.0
                month = date.month
                
                # Ramadan effect (approximate)
                if month in [3, 4]:
                    special_multiplier *= 1.5
                
                # Black Friday / Cyber Monday (November)
                if month == 11 and date.day in range(20, 30):
                    special_multiplier *= 2.0
                
                # National Day celebrations
                if (store['country'] == 'UAE' and month == 12 and date.day == 2) or \
                   (store['country'] == 'Saudi Arabia' and month == 9 and date.day == 23):
                    special_multiplier *= 1.8
                
                # Promotional periods (random)
                if random.random() < 0.1:  # 10% of days have promotions
                    promotion_flag = 1
                    promotion_discount = random.uniform(0.1, 0.4)
                    promotion_multiplier = 1 + promotion_discount * 2  # Sales boost
                else:
                    promotion_flag = 0
                    promotion_discount = 0
                    promotion_multiplier = 1.0
                
                # Calculate sales
                expected_sales = (base_sales * trend * annual_seasonality * 
                                 weekly_seasonality * special_multiplier * 
                                 promotion_multiplier)
                
                # Add noise
                noise = np.random.normal(1, 0.15)  # 15% noise
                actual_sales = max(0, expected_sales * noise)
                
                # Number of transactions
                avg_transaction_value = brand['avg_price_point'] * random.uniform(0.8, 1.2)
                num_transactions = max(0, int(actual_sales / avg_transaction_value))
                
                sale_record = {
                    'date': date.strftime('%Y-%m-%d'),
                    'store_id': store['store_id'],
                    'brand_id': store['brand_id'],
                    'country': store['country'],
                    'sales_amount': round(actual_sales, 2),
                    'num_transactions': num_transactions,
                    'avg_transaction_value': round(avg_transaction_value, 2),
                    'promotion_flag': promotion_flag,
                    'promotion_discount': round(promotion_discount, 2),
                    'day_of_week': day_of_week,
                    'is_weekend': 1 if day_of_week >= 5 else 0,
                    'month': month,
                    'year': date.year
                }
                
                all_sales.append(sale_record)
        
        df = pd.DataFrame(all_sales)
        print(f" Created {len(df):,} sales records")
        return df
    
    def generate_external_features(self):
        """Generate external features (weather, economic indicators, etc.)"""
        print("\n Generating external features...")
        dates = pd.date_range(
            start=self.end_date - timedelta(days=self.max_history_days),
            end=self.end_date,
            freq='D'
        )
        
        external_data = []
        
        for country in self.countries:
            for date in dates:
                # Economic indicators (monthly granularity, interpolated daily)
                month_offset = (date.year - 2022) * 12 + date.month
                gdp_growth = 3.0 + np.sin(month_offset / 6) * 0.5
                
                # Tourism stats (seasonal)
                day_of_year = date.timetuple().tm_yday
                tourism_index = 100 + 30 * np.sin(2 * np.pi * (day_of_year - 180) / 365)
                
                # Currency rate (slight fluctuation)
                base_rate = {'UAE': 3.67, 'Saudi Arabia': 3.75, 'Kuwait': 0.31,
                            'Qatar': 3.64, 'Bahrain': 0.38, 'Oman': 0.38,
                            'Egypt': 30.0, 'Jordan': 0.71}[country]
                currency_rate = base_rate * (1 + np.random.normal(0, 0.02))
                
                # Weather (temperature in Celsius)
                avg_temp = 25 + 15 * np.sin(2 * np.pi * (day_of_year - 30) / 365)
                temp = avg_temp + np.random.normal(0, 3)
                
                record = {
                    'date': date.strftime('%Y-%m-%d'),
                    'country': country,
                    'gdp_growth': round(gdp_growth, 2),
                    'tourism_index': round(tourism_index, 1),
                    'currency_rate_to_usd': round(currency_rate, 4),
                    'temperature_celsius': round(temp, 1),
                    'is_public_holiday': 1 if random.random() < 0.05 else 0
                }
                
                external_data.append(record)
        
        df = pd.DataFrame(external_data)
        print(f" Created {len(df):,} external feature records")
        return df
    
    def generate_all_data(self):
        """Generate all synthetic datasets"""
        print("\n" + "="*70)
        print(" STARTING DATA GENERATION")
        print("="*70)
        
        # Generate metadata
        stores_df = self.generate_store_metadata()
        brands_df = self.generate_brand_metadata()
        
        # Generate transactional data
        sales_df = self.generate_sales_data(stores_df, brands_df)
        
        # Generate external features
        external_df = self.generate_external_features()
        
        print("\n" + "="*70)
        print(" DATA GENERATION COMPLETE!")
        print("="*70)
        print(f"\nSummary:")
        print(f"  • Stores: {len(stores_df):,}")
        print(f"  • Brands: {len(brands_df):,}")
        print(f"  • Sales Records: {len(sales_df):,}")
        print(f"  • External Features: {len(external_df):,}")
        print(f"  • Date Range: {sales_df['date'].min()} to {sales_df['date'].max()}")
        print("="*70)
        
        return {
            'stores': stores_df,
            'brands': brands_df,
            'sales': sales_df,
            'external': external_df
        }
    
    def save_data(self, data_dict, output_dir='data'):
        """Save all datasets to CSV files"""
        print(f"\n Saving data to '{output_dir}' folder...")
        os.makedirs(output_dir, exist_ok=True)
        
        for name, df in data_dict.items():
            filepath = os.path.join(output_dir, f'{name}.csv')
            df.to_csv(filepath, index=False)
            print(f"  ✓ Saved {filepath} ({len(df):,} rows)")
        
        # Save summary stats
        summary = {
            'generated_at': datetime.now().isoformat(),
            'total_stores': len(data_dict['stores']),
            'total_brands': len(data_dict['brands']),
            'total_sales_records': len(data_dict['sales']),
            'date_range': {
                'start': data_dict['sales']['date'].min(),
                'end': data_dict['sales']['date'].max()
            },
            'data_tiers': data_dict['stores']['tier'].value_counts().to_dict(),
            'countries': data_dict['stores']['country'].value_counts().to_dict(),
            'brands_by_category': data_dict['brands']['category'].value_counts().to_dict()
        }
        
        with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"  ✓ Saved summary.json")
        print(f"\n All data saved successfully to '{output_dir}/' folder!")


if __name__ == '__main__':
    print("\n")
    print("--------------------------------------------------------------------")
    print("~~   SALES FORECASTING PLATFORM - SYNTHETIC DATA GENERATION         ~")
    print("~~   Based on Chalhoub Group Scaling to 500 Stores Use Case         ")
    print("--------------------------------------------------------------------")
    print("\n")
    
    generator = SyntheticDataGenerator()
    data = generator.generate_all_data()
    generator.save_data(data)
    
    print("\n" + "="*70)
    print(" READY TO BUILD THE FORECASTING PLATFORM!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Check the 'data' folder for all CSV files")
    print("  2. Run the pipeline modules in order:")
    print("     • Pipeline 1: Data Ingestion")
    print("     • Pipeline 2: Model Training")
    print("     • Pipeline 3: Inference")
    print("     • Pipeline 4: Monitoring")
    print("="*70)
    print("\n")
