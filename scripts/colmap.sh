scan="scene0050_00"

colmap feature_extractor --database_path dataset/colmap/${scan}/database.db \
    --image_path dataset/colmap/${scan}/image

colmap exhaustive_matcher --database_path dataset/colmap/${scan}/database.db

mkdir dataset/colmap/${scan}/sparse

colmap mapper --database_path dataset/colmap/${scan}/database.db \
    --image_path dataset/colmap/${scan}/image \
    --output_path dataset/colmap/${scan}/sparse

mkdir dataset/colmap/${scan}/dense

colmap image_undistorter \
    --image_path dataset/colmap/${scan}/image \
    --input_path dataset/colmap/${scan}/sparse/0 \
    --output_path dataset/colmap/${scan}/dense \
    --output_type COLMAP \

colmap patch_match_stereo \
    --workspace_path dataset/colmap/${scan}/dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true

colmap stereo_fusion \
    --workspace_path dataset/colmap/${scan}/dense \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path dataset/colmap/${scan}/dense/fused.ply

colmap poisson_mesher \
    --input_path dataset/colmap/${scan}/dense/fused.ply \
    --output_path dataset/colmap/${scan}/dense/meshed-poisson.ply

colmap delaunay_mesher \
    --input_path dataset/colmap/${scan}/dense \
    --output_path dataset/colmap/${scan}/dense/meshed-delaunay.ply
