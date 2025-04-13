from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

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
        # First try with cloud filter
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(point)
            .filterDate(start_date, end_date)
            .sort("CLOUDY_PIXEL_PERCENTAGE")
        )
        # Check if collection is empty
        if collection.size().getInfo() == 0:
            # Fallback to collection with higher cloud cover
            collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(point).filterDate(start_date, end_date).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))  # Increased cloud threshold

            if collection.size().getInfo() == 0:
                # Final fallback - no cloud filter
                collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                    .filterBounds(point)
                    .filterDate(start_date, end_date)
                )

                if collection.size().getInfo() == 0:
                    return {
                        'NDVI': None,
                        'GNDVI': None,
                        'DWSI': None,
                        'RSV1': None,
                        'data_available': False
                    }

        # Calculate median image
        median_img = collection.median()

        # Cloud masking using the QA60 band (bit 10)
        qa60 = median_img.select(['QA60']).toInt()
        cloud_mask = qa60.bitwiseAnd(1 << 10).eq(0)  # Mask out clouds (if QA60 bit is 0)

        # Apply cloud mask to the median image
        cloud_free_img = median_img.updateMask(cloud_mask)

        # --- helper function ---
        def calculate_index(img, method, *args):
            try:
                if method == 'nd':
                    result = img.normalizedDifference(args).rename('nd').reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=point,
                        scale=10,
                        bestEffort=True
                    ).getInfo()
                    return result.get('nd', None)
                elif method == 'ratio':
                    result = img.select(args[0]).divide(img.select(args[1])).rename('ratio').reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=point,
                        scale=10,
                        bestEffort=True
                    ).getInfo()
                    return result.get('ratio', None)
            except Exception as e:
                print(f"Error calculating index: {e}")
                return None

        # --- Final return with calculated indices ---
        return {
            'NDVI': calculate_index(cloud_free_img, 'nd', 'B8', 'B4'),
            'GNDVI': calculate_index(cloud_free_img, 'nd', 'B8', 'B3'),
            'DWSI': calculate_index(cloud_free_img, 'nd', 'B8', 'B11'),
            'RSV1': calculate_index(cloud_free_img, 'ratio', 'B4', 'B2'),
            'data_available': True
        }

    except Exception as e:
        logging.error(f"Error in get_indices_with_fallback: {str(e)}")
        return {
            'NDVI': None,
            'GNDVI': None,
            'DWSI': None,
            'RSV1': None,
            'data_available': False
        }

@app.route('/monitoring', methods=['POST'])
def monitoring_graph():
    try:
        data = request.get_json()

        # Get polygon coordinates
        coordinates = data.get('coordinates')
        if not coordinates:
            return jsonify({'error': 'Missing polygon coordinates'}), 400

        # Parse start_date
        start_date_str = data.get('start_date')
        if not start_date_str:
            return jsonify({'error': 'Missing start_date'}), 400
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        except ValueError:
            return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD.'}), 400

        # Optional: You may want to extract centroid for lat/lon-based fallback
        polygon = ee.Geometry.Polygon(coordinates)
        centroid = polygon.centroid().coordinates().getInfo()
        lon, lat = centroid[0], centroid[1]

    except Exception as e:
        return jsonify({'error': str(e)}), 400

    # Prepare GEE data processing
    try:
        response_data = []
        missing_data_count = 0

        # 7-day intervals from day 0–125, 14-day intervals from day 127–177
        time_windows = (
            [(start_date + timedelta(days=i), 7) for i in range(0, 126, 7)] +
            [(start_date + timedelta(days=i), 14) for i in range(127, 178, 7)]
        )
        for begin, days in time_windows:
            end = begin + timedelta(days=days)
            try:
                indices = get_indices(polygon, begin.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))

                if indices and indices.get('data_available') and any([
                    indices.get('NDVI') is not None,
                    indices.get('GNDVI') is not None,
                    indices.get('DWSI') is not None,
                    indices.get('RSV1') is not None
                ]):
                    response_data.append({
                        'date': begin.strftime('%Y-%m-%d'),
                        'NDVI': indices.get('NDVI'),
                        'GNDVI': indices.get('GNDVI'),
                        'DWSI': indices.get('DWSI'),
                        'RSV1': indices.get('RSV1'),
                        'data_available': True
                    })
                else:
                    missing_data_count += 1
            except Exception as e:
                print(e)
                missing_data_count += 1
                response_data.append({
                    'date': begin.strftime('%Y-%m-%d'),
                    'error': str(e),
                    'data_available': False
                })

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
        .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']))
        image = collection.median()
        image = image.divide(10000)
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        gndvi = image.normalizedDifference(['B8', 'B3']).rename('GNDVI')
        npc_i = image.normalizedDifference(['B4', 'B2']).rename('NPCI')
        dwsi = image.expression('(B8A - B11) / (B8A + B11)', {
            'B8A': image.select('B8A'),
            'B11': image.select('B11')
        }).rename('DWSI')
        rvsi = image.expression('(B3 - B2) / (B3 + B2)', {
            'B2': image.select('B2'),
            'B3': image.select('B3')
        }).rename('RVSI')


        # Combine all bands and indices into one image
        combined = image.addBands([ndvi, gndvi, npc_i, dwsi, rvsi])

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