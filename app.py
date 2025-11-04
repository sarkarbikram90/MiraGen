import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import string
import io
import json
from faker import Faker

# Page configuration
st.set_page_config(
    page_title="Custom Data Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Faker
fake = Faker()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .column-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e5e7eb;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .delete-btn {
        background-color: #ef4444;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        cursor: pointer;
    }
    
    .success-box {
        background-color: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Data type options with their generators
DATA_TYPES = {
    "Text": {
        "Random Text": lambda: fake.text(max_nb_chars=50),
        "Word": lambda: fake.word(),
        "Sentence": lambda: fake.sentence(),
        "Paragraph": lambda: fake.paragraph(),
        "Name": lambda: fake.name(),
        "First Name": lambda: fake.first_name(),
        "Last Name": lambda: fake.last_name(),
        "Company": lambda: fake.company(),
        "Job Title": lambda: fake.job(),
        "Email": lambda: fake.email(),
        "Username": lambda: fake.user_name(),
        "Domain": lambda: fake.domain_name(),
        "URL": lambda: fake.url(),
        "IPv4": lambda: fake.ipv4(),
        "IPv6": lambda: fake.ipv6(),
        "User Agent": lambda: fake.user_agent(),
        "Custom List": None  # Special handling
    },
    "Number": {
        "Integer": None,  # Requires range
        "Float": None,    # Requires range
        "Percentage": lambda: round(random.uniform(0, 100), 2),
        "Currency": None, # Requires range
        "Phone Number": lambda: fake.phone_number(),
        "Credit Card": lambda: fake.credit_card_number(),
        "SSN": lambda: fake.ssn(),
    },
    "Date/Time": {
        "Date": None,           # Requires range
        "DateTime": None,       # Requires range
        "Time": lambda: fake.time(),
        "Past Date": lambda: fake.date_between(start_date='-5y', end_date='today').strftime('%Y-%m-%d'),
        "Future Date": lambda: fake.date_between(start_date='today', end_date='+5y').strftime('%Y-%m-%d'),
        "Birth Date": lambda: fake.date_of_birth(minimum_age=18, maximum_age=80).strftime('%Y-%m-%d'),
        "Timestamp": lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    },
    "Location": {
        "Address": lambda: fake.address().replace('\n', ', '),
        "Street Address": lambda: fake.street_address(),
        "City": lambda: fake.city(),
        "State": lambda: fake.state(),
        "Country": lambda: fake.country(),
        "Country Code": lambda: fake.country_code(),
        "Zipcode": lambda: fake.zipcode(),
        "Latitude": lambda: fake.latitude(),
        "Longitude": lambda: fake.longitude(),
        "Coordinates": lambda: f"{fake.latitude()}, {fake.longitude()}",
    },
    "Boolean": {
        "True/False": lambda: random.choice([True, False]),
        "Yes/No": lambda: random.choice(['Yes', 'No']),
        "1/0": lambda: random.choice([1, 0]),
        "Active/Inactive": lambda: random.choice(['Active', 'Inactive']),
    },
    "ID/UUID": {
        "UUID4": lambda: fake.uuid4(),
        "Sequential ID": None,  # Special handling
        "Random ID": lambda: ''.join(random.choices(string.ascii_uppercase + string.digits, k=10)),
        "Hex ID": lambda: fake.hexify(text='^^^^^^^^'),
    }
}

def initialize_session_state():
    """Initialize session state variables"""
    if 'columns' not in st.session_state:
        st.session_state.columns = []
    if 'generated_data' not in st.session_state:
        st.session_state.generated_data = None
    if 'column_counter' not in st.session_state:
        st.session_state.column_counter = 0

def add_column():
    """Add a new column to the schema"""
    st.session_state.column_counter += 1
    new_column = {
        'id': st.session_state.column_counter,
        'name': f'Column_{st.session_state.column_counter}',
        'data_category': 'Text',
        'data_type': 'Random Text',
        'config': {}
    }
    st.session_state.columns.append(new_column)

def remove_column(column_id):
    """Remove a column from the schema"""
    st.session_state.columns = [col for col in st.session_state.columns if col['id'] != column_id]

def generate_value(column):
    """Generate a single value based on column configuration"""
    category = column['data_category']
    data_type = column['data_type']
    config = column.get('config', {})
    
    # Handle special cases
    if data_type == "Custom List" and 'custom_values' in config:
        values = [v.strip() for v in config['custom_values'].split(',') if v.strip()]
        return random.choice(values) if values else "N/A"
    
    elif data_type == "Integer":
        min_val = config.get('min_value', 0)
        max_val = config.get('max_value', 100)
        return random.randint(min_val, max_val)
    
    elif data_type == "Float":
        min_val = config.get('min_value', 0.0)
        max_val = config.get('max_value', 100.0)
        decimals = config.get('decimals', 2)
        return round(random.uniform(min_val, max_val), decimals)
    
    elif data_type == "Currency":
        min_val = config.get('min_value', 0.0)
        max_val = config.get('max_value', 10000.0)
        symbol = config.get('currency_symbol', '$')
        value = round(random.uniform(min_val, max_val), 2)
        return f"{symbol}{value:,.2f}"
    
    elif data_type == "Date":
        start_date = config.get('start_date', datetime.now() - timedelta(days=365))
        end_date = config.get('end_date', datetime.now())
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        random_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
        return random_date.strftime('%Y-%m-%d')
    
    elif data_type == "DateTime":
        start_date = config.get('start_date', datetime.now() - timedelta(days=365))
        end_date = config.get('end_date', datetime.now())
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        random_date = start_date + timedelta(
            days=random.randint(0, (end_date - start_date).days),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
        return random_date.strftime('%Y-%m-%d %H:%M:%S')
    
    elif data_type == "Sequential ID":
        # This will be handled specially during generation
        return None
    
    # Default: use the generator function
    generator = DATA_TYPES.get(category, {}).get(data_type)
    if generator and callable(generator):
        return generator()
    
    return "N/A"

def generate_dataset(num_records):
    """Generate the complete dataset"""
    if not st.session_state.columns:
        st.error("❌ Please add at least one column to generate data!")
        return None
    
    data = {}
    
    # Initialize data dictionary
    for column in st.session_state.columns:
        data[column['name']] = []
    
    # Generate data
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(num_records):
        for column in st.session_state.columns:
            if column['data_type'] == "Sequential ID":
                value = i + 1
            else:
                value = generate_value(column)
            data[column['name']].append(value)
        
        # Update progress
        if (i + 1) % 100 == 0 or i == num_records - 1:
            progress = (i + 1) / num_records
            progress_bar.progress(progress)
            status_text.text(f"Generating records... {i + 1}/{num_records}")
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(data)

def render_column_config(column, index):
    """Render configuration UI for a column"""
    st.markdown(f"### Column {index + 1}")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        column['name'] = st.text_input(
            "Column Name",
            value=column['name'],
            key=f"name_{column['id']}"
        )
    
    with col2:
        if st.button("🗑️ Delete", key=f"delete_{column['id']}", type="secondary"):
            remove_column(column['id'])
            st.rerun()
    
    col1, col2 = st.columns(2)
    
    with col1:
        column['data_category'] = st.selectbox(
            "Data Category",
            options=list(DATA_TYPES.keys()),
            index=list(DATA_TYPES.keys()).index(column['data_category']),
            key=f"category_{column['id']}"
        )
    
    with col2:
        data_types_in_category = list(DATA_TYPES[column['data_category']].keys())
        current_index = 0
        if column['data_type'] in data_types_in_category:
            current_index = data_types_in_category.index(column['data_type'])
        
        column['data_type'] = st.selectbox(
            "Data Type",
            options=data_types_in_category,
            index=current_index,
            key=f"type_{column['id']}"
        )
    
    # Additional configuration based on data type
    if column['data_type'] == "Custom List":
        column['config']['custom_values'] = st.text_area(
            "Enter values (comma-separated)",
            value=column['config'].get('custom_values', ''),
            key=f"custom_{column['id']}",
            help="Example: Red, Green, Blue, Yellow"
        )
    
    elif column['data_type'] == "Integer":
        col1, col2 = st.columns(2)
        with col1:
            column['config']['min_value'] = st.number_input(
                "Min Value",
                value=column['config'].get('min_value', 0),
                key=f"min_{column['id']}"
            )
        with col2:
            column['config']['max_value'] = st.number_input(
                "Max Value",
                value=column['config'].get('max_value', 100),
                key=f"max_{column['id']}"
            )
    
    elif column['data_type'] == "Float":
        col1, col2, col3 = st.columns(3)
        with col1:
            column['config']['min_value'] = st.number_input(
                "Min Value",
                value=float(column['config'].get('min_value', 0.0)),
                key=f"min_{column['id']}"
            )
        with col2:
            column['config']['max_value'] = st.number_input(
                "Max Value",
                value=float(column['config'].get('max_value', 100.0)),
                key=f"max_{column['id']}"
            )
        with col3:
            column['config']['decimals'] = st.number_input(
                "Decimal Places",
                value=column['config'].get('decimals', 2),
                min_value=0,
                max_value=10,
                key=f"decimals_{column['id']}"
            )
    
    elif column['data_type'] == "Currency":
        col1, col2, col3 = st.columns(3)
        with col1:
            column['config']['currency_symbol'] = st.text_input(
                "Currency Symbol",
                value=column['config'].get('currency_symbol', '$'),
                key=f"symbol_{column['id']}"
            )
        with col2:
            column['config']['min_value'] = st.number_input(
                "Min Value",
                value=float(column['config'].get('min_value', 0.0)),
                key=f"min_{column['id']}"
            )
        with col3:
            column['config']['max_value'] = st.number_input(
                "Max Value",
                value=float(column['config'].get('max_value', 10000.0)),
                key=f"max_{column['id']}"
            )
    
    elif column['data_type'] in ["Date", "DateTime"]:
        col1, col2 = st.columns(2)
        with col1:
            default_start = datetime.now() - timedelta(days=365)
            column['config']['start_date'] = st.date_input(
                "Start Date",
                value=datetime.strptime(column['config'].get('start_date', default_start.strftime('%Y-%m-%d')), '%Y-%m-%d') if isinstance(column['config'].get('start_date'), str) else column['config'].get('start_date', default_start),
                key=f"start_{column['id']}"
            ).strftime('%Y-%m-%d')
        with col2:
            default_end = datetime.now()
            column['config']['end_date'] = st.date_input(
                "End Date",
                value=datetime.strptime(column['config'].get('end_date', default_end.strftime('%Y-%m-%d')), '%Y-%m-%d') if isinstance(column['config'].get('end_date'), str) else column['config'].get('end_date', default_end),
                key=f"end_{column['id']}"
            ).strftime('%Y-%m-%d')
    
    st.markdown("---")

def main():
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🎨 Custom Synthetic Data Generator</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; color: #666; font-size: 1.1rem; margin-bottom: 2rem;">'
        'Design your own dataset schema and generate custom synthetic data'
        '</p>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    st.sidebar.header("⚙️ Generation Settings")
    
    num_records = st.sidebar.slider(
        "Number of Records",
        min_value=10,
        max_value=50000,
        value=1000,
        step=10,
        help="How many rows to generate"
    )
    
    st.sidebar.markdown("---")
    
    # Export format
    export_format = st.sidebar.multiselect(
        "Export Formats",
        options=["CSV", "JSON", "Excel"],
        default=["CSV"],
        help="Select one or more export formats"
    )
    
    st.sidebar.markdown("---")
    
    # Tips
    with st.sidebar.expander("💡 Quick Tips"):
        st.markdown("""
        **Getting Started:**
        1. Click 'Add Column' to create a new field
        2. Give it a meaningful name
        3. Choose data category and type
        4. Configure additional settings
        5. Click 'Generate Data'
        
        **Pro Tips:**
        - Use Sequential ID for unique identifiers
        - Custom List for predefined values
        - Date ranges for time-series data
        """)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📋 Schema Builder", "📊 Generated Data", "💾 Templates"])
    
    with tab1:
        st.markdown("## Define Your Data Schema")
        
        # Add column button
        if st.button("➕ Add Column", type="primary", use_container_width=True):
            add_column()
            st.rerun()
        
        st.markdown("---")
        
        # Display columns
        if st.session_state.columns:
            for idx, column in enumerate(st.session_state.columns):
                with st.container():
                    render_column_config(column, idx)
        else:
            st.info("👆 Click 'Add Column' to start building your dataset schema")
        
        # Generate button
        if st.session_state.columns:
            st.markdown("---")
            if st.button("🚀 Generate Data", type="primary", use_container_width=True):
                with st.spinner("Generating your custom dataset..."):
                    df = generate_dataset(num_records)
                    if df is not None:
                        st.session_state.generated_data = df
                        st.success(f"✅ Successfully generated {len(df):,} records with {len(df.columns)} columns!")
                        st.balloons()
    
    with tab2:
        if st.session_state.generated_data is not None:
            df = st.session_state.generated_data
            
            # Statistics
            st.markdown("### 📈 Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", f"{len(df):,}")
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
                st.metric("Memory Usage", f"{memory_mb:.2f} MB")
            with col4:
                st.metric("Data Types", df.dtypes.nunique())
            
            st.markdown("---")
            
            # Download section
            st.markdown("### 💾 Download Your Data")
            
            download_cols = st.columns(len(export_format) if export_format else 1)
            
            for idx, fmt in enumerate(export_format):
                with download_cols[idx]:
                    if fmt == "CSV":
                        csv_buffer = io.StringIO()
                        df.to_csv(csv_buffer, index=False)
                        st.download_button(
                            label="📄 Download CSV",
                            data=csv_buffer.getvalue(),
                            file_name=f"custom_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    elif fmt == "JSON":
                        json_data = df.to_json(orient='records', indent=2)
                        st.download_button(
                            label="📋 Download JSON",
                            data=json_data,
                            file_name=f"custom_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    
                    elif fmt == "Excel":
                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                            df.to_excel(writer, sheet_name='Custom Data', index=False)
                        st.download_button(
                            label="📊 Download Excel",
                            data=excel_buffer.getvalue(),
                            file_name=f"custom_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
            
            st.markdown("---")
            
            # Data preview
            st.markdown("### 🔍 Data Preview")
            
            # Column info
            with st.expander("📋 Column Information"):
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes.astype(str),
                    'Non-Null': df.count().values,
                    'Unique Values': df.nunique().values,
                    'Sample Value': [df[col].iloc[0] if len(df) > 0 else 'N/A' for col in df.columns]
                })
                st.dataframe(col_info, use_container_width=True)
            
            # Display data
            st.dataframe(df.head(50), use_container_width=True, height=400)
            
            if len(df) > 50:
                st.info(f"📌 Showing first 50 rows out of {len(df):,} total records")
        
        else:
            st.info("👈 Generate data from the Schema Builder tab first")
    
    with tab3:
        st.markdown("## 📦 Schema Templates")
        st.markdown("Quick start with pre-built templates")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("👤 Customer Profile Template", use_container_width=True):
                st.session_state.columns = [
                    {'id': 1, 'name': 'Customer_ID', 'data_category': 'ID/UUID', 'data_type': 'Sequential ID', 'config': {}},
                    {'id': 2, 'name': 'First_Name', 'data_category': 'Text', 'data_type': 'First Name', 'config': {}},
                    {'id': 3, 'name': 'Last_Name', 'data_category': 'Text', 'data_type': 'Last Name', 'config': {}},
                    {'id': 4, 'name': 'Email', 'data_category': 'Text', 'data_type': 'Email', 'config': {}},
                    {'id': 5, 'name': 'Phone', 'data_category': 'Number', 'data_type': 'Phone Number', 'config': {}},
                    {'id': 6, 'name': 'City', 'data_category': 'Location', 'data_type': 'City', 'config': {}},
                    {'id': 7, 'name': 'State', 'data_category': 'Location', 'data_type': 'State', 'config': {}},
                    {'id': 8, 'name': 'Join_Date', 'data_category': 'Date/Time', 'data_type': 'Past Date', 'config': {}},
                    {'id': 9, 'name': 'Status', 'data_category': 'Boolean', 'data_type': 'Active/Inactive', 'config': {}},
                ]
                st.session_state.column_counter = 9
                st.rerun()
            
            if st.button("💳 Transaction Template", use_container_width=True):
                st.session_state.columns = [
                    {'id': 1, 'name': 'Transaction_ID', 'data_category': 'ID/UUID', 'data_type': 'UUID4', 'config': {}},
                    {'id': 2, 'name': 'Customer_ID', 'data_category': 'ID/UUID', 'data_type': 'Sequential ID', 'config': {}},
                    {'id': 3, 'name': 'Amount', 'data_category': 'Number', 'data_type': 'Currency', 'config': {'min_value': 10, 'max_value': 5000, 'currency_symbol': '$'}},
                    {'id': 4, 'name': 'Transaction_Date', 'data_category': 'Date/Time', 'data_type': 'DateTime', 'config': {}},
                    {'id': 5, 'name': 'Status', 'data_category': 'Text', 'data_type': 'Custom List', 'config': {'custom_values': 'Completed, Pending, Failed, Refunded'}},
                    {'id': 6, 'name': 'Payment_Method', 'data_category': 'Text', 'data_type': 'Custom List', 'config': {'custom_values': 'Credit Card, Debit Card, PayPal, Bank Transfer'}},
                ]
                st.session_state.column_counter = 6
                st.rerun()
        
        with col2:
            if st.button("👨‍💼 Employee Template", use_container_width=True):
                st.session_state.columns = [
                    {'id': 1, 'name': 'Employee_ID', 'data_category': 'ID/UUID', 'data_type': 'Sequential ID', 'config': {}},
                    {'id': 2, 'name': 'Full_Name', 'data_category': 'Text', 'data_type': 'Name', 'config': {}},
                    {'id': 3, 'name': 'Email', 'data_category': 'Text', 'data_type': 'Email', 'config': {}},
                    {'id': 4, 'name': 'Department', 'data_category': 'Text', 'data_type': 'Custom List', 'config': {'custom_values': 'Engineering, Sales, Marketing, HR, Finance, Operations'}},
                    {'id': 5, 'name': 'Job_Title', 'data_category': 'Text', 'data_type': 'Job Title', 'config': {}},
                    {'id': 6, 'name': 'Salary', 'data_category': 'Number', 'data_type': 'Integer', 'config': {'min_value': 40000, 'max_value': 200000}},
                    {'id': 7, 'name': 'Hire_Date', 'data_category': 'Date/Time', 'data_type': 'Past Date', 'config': {}},
                    {'id': 8, 'name': 'Active', 'data_category': 'Boolean', 'data_type': 'True/False', 'config': {}},
                ]
                st.session_state.column_counter = 8
                st.rerun()
            
            if st.button("🚨 IT Alert Template", use_container_width=True):
                st.session_state.columns = [
                    {'id': 1, 'name': 'Alert_ID', 'data_category': 'ID/UUID', 'data_type': 'UUID4', 'config': {}},
                    {'id': 2, 'name': 'Timestamp', 'data_category': 'Date/Time', 'data_type': 'DateTime', 'config': {}},
                    {'id': 3, 'name': 'System_Source', 'data_category': 'Text', 'data_type': 'Custom List', 'config': {'custom_values': 'SAP_ECC, HANA_DB, BTP_Tenant, App_Server, API_Gateway'}},
                    {'id': 4, 'name': 'Category', 'data_category': 'Text', 'data_type': 'Custom List', 'config': {'custom_values': 'Performance, Availability, Exception, Database, Security'}},
                    {'id': 5, 'name': 'Priority', 'data_category': 'Text', 'data_type': 'Custom List', 'config': {'custom_values': 'P1, P2, P3, P4, P5'}},
                    {'id': 6, 'name': 'Status', 'data_category': 'Text', 'data_type': 'Custom List', 'config': {'custom_values': 'Open, Acknowledged, Resolved'}},
                    {'id': 7, 'name': 'Severity_Score', 'data_category': 'Number', 'data_type': 'Integer', 'config': {'min_value': 0, 'max_value': 100}},
                ]
                st.session_state.column_counter = 7
                st.rerun()
        
        st.markdown("---")
        st.info("💡 **Tip:** Load a template, then customize it to your needs in the Schema Builder tab!")

    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #999; padding: 1rem;">'
        '🎨 Custom Data Generator | Build any dataset you can imagine'
        '</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()