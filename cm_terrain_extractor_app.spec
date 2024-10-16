# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['cm_terrain_extractor_app/cm_terrain_extractor_app.py'],
    pathex=[],
    binaries=[('C:/Users/der_b/miniconda3/envs/cm_terrain/Library/bin/gdal.dll', 'dll')],
    datas=[
        (
            "C:/Users/der_b/miniconda3/envs/cm_terrain/Lib/site-packages/streamlit/static",
            "./streamlit/static"
        ),
        (
            "C:/Users/der_b/miniconda3/envs/cm_terrain/Lib/site-packages/streamlit_folium/frontend/build/",
            "./streamlit_folium/frontend/build/"
        ),
                

        (   
            "E:/Spiele/CMAutoEditor/cm_terrain_extractor_app",
            "./cm_terrain_extractor_app"
        ),

        (
            "E:/Spiele/CMAutoEditor/profiles",
            "./profiles"
        ),

    ],    
    hiddenimports=['streamlit', 'streamlit_folium', 'shapely', 'pyproj', 'geopandas', 'rasterio', 'py7zr', 'rasterio.transform', 'rasterio.sample', 'rasterio.vrt', 
                   'rasterio._features', 'rasterio.fill', 'rasterio.merge', "skimage.measure", "skimage.transform", 'rasterio.warp', 
                   'terrain_extraction.projection_utils', 'terrain_extraction.bbox_utils', 'terrain_extraction.data_sources.hessen_dgm1.data_source', 
                   'terrain_extraction.data_sources.aw3d30.data_source', 'terrain_extraction.data_sources.nrw_dgm1.data_source', 'terrain_extraction.data_sources.netherlands_dtm05.data_source',
                   'terrain_extraction.data_sources.bavaria_dgm1.data_source', 'terrain_extraction.data_sources.thuringia_dgm1.data_source',
                   'terrain_extraction.elevation_map', 'terrain_extraction.data_sources.rge_alti.data_source', 'fiona._shim', 'rasterio._shim', 'osmnx', 'geojson', 'profiles'],

    hookspath=['./hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,

)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='cm_terrain_extractor_app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
