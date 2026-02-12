import numpy as np
import spheni

def main():
    # 1. Create an index with Haversine metric
    #    Note: dimension MUST be 2 (latitude, longitude)
    #    Note: normalization MUST be False
    spec = spheni.IndexSpec(2, spheni.Metric.Haversine, spheni.IndexKind.Flat)
    engine = spheni.Engine(spec)

    # 2. Add some locations (Latitude, Longitude)
    #    New York, London, Paris, Tokyo, Sydney
    locations = np.array([
        [40.7128, -74.0060],   # NY
        [51.5074, -0.1278],    # London
        [48.8566, 2.3522],     # Paris
        [35.6762, 139.6503],   # Tokyo
        [-33.8688, 151.2093]   # Sydney
    ], dtype=np.float32)
    
    ids = np.array([0, 1, 2, 3, 4], dtype=np.longlong)
    names = ["NY", "London", "Paris", "Tokyo", "Sydney"]
    
    engine.add(ids, locations)
    print(f"Indexed {engine.size()} locations.")

    # 3. Search for nearest neighbors to Berlin (52.5200, 13.4050)
    query = np.array([52.5200, 13.4050], dtype=np.float32)
    k = 3
    results = engine.search(query, k)

    print(f"\nFinding {k} nearest cities to Berlin:")
    for hit in results:
        # Haversine metric returns negative distance in kilometers
        dist_km = -hit.score
        name = names[hit.id]
        print(f"- {name}: {dist_km:.2f} km")

if __name__ == "__main__":
    main()
