from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor


import ee, os
from google.oauth2 import service_account
import google.auth.transport.requests
# Authenticate using service account
#SERVICE_ACCOUNT_FILE = "certain-catcher-430110-v2-7beec7335614.json"
#SCOPES = ["https://www.googleapis.com/auth/earthengine.readonly"]

#credentials = service_account.Credentials.from_service_account_file(
#    SERVICE_ACCOUNT_FILE,
#    scopes=SCOPES
#)

#ee.Initialize(credentials)
app = Flask(__name__)
model = joblib.load('model.pkl')

GEE_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
credentials = service_account.Credentials.from_service_account_file(GEE_CREDENTIALS,scopes=['https://www.googleapis.com/auth/earthengine'])
credentials.refresh(google.auth.transport.requests.Request())
ee.Initialize(credentials)





#ee.Initialize(project="certain-catcher-430110-v2")
def get_indices(point, start_date, end_date):
    try:
        def mask_s2_sr(image):
            # Select QA60 band and SCL (Scene Classification Layer)
            qa60 = image.select('QA60')
            scl = image.select('SCL')

            # Bit 10 and 11 in QA60 are clouds and cirrus
            cloud_bit_mask = 1 << 10  # clouds
            cirrus_bit_mask = 1 << 11  # cirrus

            # QA60 mask
            mask_qa60 = qa60.bitwiseAnd(cloud_bit_mask).eq(0).And(
                        qa60.bitwiseAnd(cirrus_bit_mask).eq(0))

            # Mask out clouds, shadows, snow using SCL values
            # Keep only classes like vegetation (4, 5), bare soils (6), water (8)
            mask_scl = scl.neq(3).And(  # cloud shadow
                        scl.neq(9)).And(  # clouds
                        scl.neq(10)).And(  # cirrus
                        scl.neq(1))  # saturated or defective

            # Combine both masks
            return image.updateMask(mask_qa60).updateMask(mask_scl)

        # Apply cloud filtering and masking
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(point)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
            .map(mask_s2_sr)  # ⬅️ apply the improved cloud mask
        )
        # Check if collection is empty
        if collection.size().getInfo() == 0:
            return {
                'NDVI': None,
                'GNDVI': None,
                'DWSI': None,
                'RVS1': None,
                'data_available': False
            }

        # Use median of cloud-free images
        cloud_free_img = collection.median()
        cloud_free_img = cloud_free_img.divide(10000)
        ndvi = cloud_free_img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        # NIR−[540:570] / NIR+[540:570]
        gndvi = cloud_free_img.normalizedDifference(['B8', 'B3']).rename('GNDVI')
        npci = cloud_free_img.expression('((B4 - B1) / (B4 + B1))', {
            'B1': cloud_free_img.select('B1'),
            'B4': cloud_free_img.select('B4')
        }).rename('NPCI')
        # 802nm+547nm / 1657nm+682nm (B8 + B3) / (B11 + B4)
        dwsi = cloud_free_img.expression('((B8 + B3) / (B11 + B4))', {
            'B3': cloud_free_img.select('B3'),
            'B4': cloud_free_img.select('B4'),
            'B8': cloud_free_img.select('B8'),
            'B11': cloud_free_img.select('B11')
        }).rename('DWSI')
        #718nm+748nm/2−733nm
        rvs1 = cloud_free_img.expression('((B5 + B7) / 2) - B6', {
            'B5': cloud_free_img.select('B5'),
            'B6': cloud_free_img.select('B6'),
            'B7': cloud_free_img.select('B7')
        }).rename('RVS1')

        all_indices = ndvi.addBands(gndvi).addBands(dwsi).addBands(rvs1)

        values = all_indices.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=10,
            bestEffort=True
        ).getInfo()
        return {
            'NDVI': values.get('NDVI'),
            'GNDVI': values.get('GNDVI'),
            'DWSI': values.get('DWSI'),
            'RVS1': values.get('RVS1'),
            'data_available': any(values.values())
        }

    except Exception as e:
        logging.error(f"Error in get_indices_with_fallback: {str(e)}")
        return {
            'NDVI': None,
            'GNDVI': None,
            'DWSI': None,
            'RVS1': None,
            'data_available': False
        }

@app.route('/monitoring', methods=['POST'])
def monitoring_graph():
    try:
        data = request.get_json()

        # Validate input
        coordinates = data.get('coordinates')
        if not coordinates:
            return jsonify({'error': 'Missing polygon coordinates'}), 400

        start_date_str = data.get('start_date')
        if not start_date_str:
            return jsonify({'error': 'Missing start_date'}), 400

        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        except ValueError:
            return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD.'}), 400

        polygon = ee.Geometry.Polygon(coordinates)

    except Exception as e:
        return jsonify({'error+++': str(e)}), 400

    time_windows = (
        [(start_date + timedelta(days=i), 5, 'Sowing') for i in range(0, 115, 5)] +  # Sowing stage
        [(start_date + timedelta(days=i), 5, 'Flowering') for i in range(115, 145, 5)] +  # Flowering stage
        [(start_date + timedelta(days=i), 5, 'Harvesting') for i in range(145, 181, 5)]  # Harvesting stage
    )

    # Helper function for parallel execution
    def process_week(week_data):
        begin, days, growth_stage = week_data
        end = begin + timedelta(days=days)
        try:
            indices = get_indices(polygon, begin, end)
            # if not indices.get('data_available', False):
                # return {
                #     'date': begin,
                #     'data_available': False,
                #     'growth_stage': growth_stage
                # }

            return {
                'date': begin,
                'NDVI': indices.get('NDVI'),
                'GNDVI': indices.get('GNDVI'),
                'DWSI': indices.get('DWSI'),
                'RVS1': indices.get('RVS1'),
                'data_available': True,
                'growth_stage': growth_stage
            }
        except Exception as e:
            return {
                'date': begin.strftime('%Y-%m-%d'),
                'error': str(e),
                'data_available': False,
                'growth':growth_stage
            }

    try:
        # Multithreading with limit
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(process_week, time_windows))

        return jsonify(results)

    except Exception as e:
        return jsonify({'error?????': str(e)}), 500

# #-------------------------------------------------------

def extract_bands_indices(geometry, start_str, end_str):
    if start_str == end_str:
        start_date = datetime.strptime(start_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_str, '%Y-%m-%d')
    try:
        point = ee.Geometry.Polygon(geometry)
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(point)
        .filterDate(ee.Date(start_date).advance(-3, 'day'), ee.Date(end_date).advance(3, 'day'))
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
        .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11']))

        image = collection.median()

        image = image.divide(10000)
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        # NIR−[540:570] / NIR+[540:570]
        gndvi = image.normalizedDifference(['B8', 'B3']).rename('GNDVI')
        npci = image.expression('((B4 - B1) / (B4 + B1))', {
            'B1': image.select('B1'),
            'B4': image.select('B4')
        }).rename('NPCI')
        # 802nm+547nm / 1657nm+682nm (B8 + B3) / (B11 + B4)
        dswi = image.expression('((B8 + B3) / (B11 + B4))', {
            'B3': image.select('B3'),
            'B4': image.select('B4'),
            'B8': image.select('B8'),
            'B11': image.select('B11')
        }).rename('DSWI')
        #718nm+748nm/2−733nm
        rvs1 = image.expression('((B5 + B7) / 2) - B6', {
            'B5': image.select('B5'),
            'B6': image.select('B6'),
            'B7': image.select('B7')
        }).rename('RVS1')


        # Combine all bands and indices into one image
        combined = image.addBands([ndvi, gndvi, npc_i, dwsi, rvs1])

        # Extract values at point
        values = combined.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=10,
            maxPixels=1e9
        ).getInfo()

        # Return values with a fallback
        return {
        'area_values': values or {},  # Fallback if empty
        'band_info': {
            # Spectral bands with exact wavelengths from your requirements
            'B2': {'name': 'Blue', 'wavelength': 496.6, 'type': 'spectral'},
            'B3': {'name': 'Green', 'wavelength': 560, 'type': 'spectral'},
            'B4': {'name': 'Red', 'wavelength': 664.5, 'type': 'spectral'},
            'B5': {'name': 'Red Edge 1', 'wavelength': 703.9, 'type': 'spectral'},
            'B6': {'name': 'Red Edge 2', 'wavelength': 740.2, 'type': 'spectral'},
            'B7': {'name': 'Red Edge 3', 'wavelength': 782.5, 'type': 'spectral'},
            'B8': {'name': 'NIR', 'wavelength': 835.1, 'type': 'spectral'},
            'B8A': {'name': 'Red Edge 4', 'wavelength': 864.8, 'type': 'spectral'},
            'B11': {'name': 'SWIR 1', 'wavelength': 1613.7, 'type': 'spectral'},
            'B12': {'name': 'SWIR 2', 'wavelength': 2202.4, 'type': 'spectral'},
            
            # Vegetation indices (no wavelength)
            'NDVI': {'name': 'NDVI', 'type': 'index'},
            'GNDVI': {'name': 'Green NDVI', 'type': 'index'},
            'NPCI': {'name': 'NPCI', 'type': 'index'},
            'DWSI': {'name': 'DWSI', 'type': 'index'},
            'RVSI': {'name': 'RVSI', 'type': 'index'}
        }
    }
    except Exception as e:
        print(end_date)
        print("Error : ", e)
        return None

@app.route('/prediction', methods=['POST'])
def prediction():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'No data provided'}), 400
        coordinates = data.get("coordinates")
        if not coordinates:
            return jsonify({'error': 'Missing polygon coordinates'}), 400
        start_date = data.get("start_date")
        if not start_date:
            return jsonify({'error': 'Missing start_date'}), 400
        end_date = data.get("end_date")
        if not end_date:
            end_date = start_date
        indices = extract_bands_indices(coordinates, start_date, end_date)
        area_values = indices['area_values']
        features = pd.DataFrame([{
        'B2 (496.6nm/492.1nm)': area_values.get('B2', 0),
        'B3 (560nm/559nm)': area_values.get('B3', 0),
        'B4 (664.5nm/665nm)': area_values.get('B4', 0),
        'B5 (703.9nm/703.8nm)': area_values.get('B5', 0),
        'B6 (740.2nm/739.1nm)': area_values.get('B6', 0),
        'B7 (782.5nm/779.7nm)': area_values.get('B7', 0),
        'B8 (835.1nm/833nm)': area_values.get('B8', 0),
        'B8A (864.8nm/864nm)': area_values.get('B8A', 0),
        'B11 (1613.7nm/1610.4nm)': area_values.get('B11', 0),
        'B12 (2202.4nm/2185.7nm)': area_values.get('B12', 0),
        'NDVI': area_values.get('NDVI', 0),
        'GNDVI': area_values.get('GNDVI', 0),
        'DWSI': area_values.get('DWSI', 0),
        'RVSI': area_values.get('RVSI', 0),
        'NPCI': area_values.get('NPCI', 0),
    }])
        print(features)
        prediction = model.predict(features)
        probabilities = model.predict_proba(features)
        result = int(prediction[0])
        return jsonify({
            "prediction": int(prediction[0]),
            "confidence": float(probabilities[0][prediction[0]]),
            "probabilities": {
                "healthy": float(probabilities[0][1]),
                "unhealthy": float(probabilities[0][0])
            },
            "band_values": indices.get('area_values', [])
        })
    except Exception as e:
        print("Error : ", e)
        return jsonify({'error': str(e)}), 500
    return jsonify({'prediction': result})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')