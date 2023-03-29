
#include <vector>
#include <array>
#include <tuple>
#include <random>
#include <numeric>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

typedef std::array<double, 3> vec3;
vec3 operator-(const vec3& first, const vec3& second) { return vec3{first[0]-second[0], first[1]-second[1], first[2]-second[2]};} 
vec3 operator+(const vec3& first, const vec3& second) { return vec3{first[0]+second[0], first[1]+second[1], first[2]+second[2]};} 


template <typename T>
inline T dot_product_3D(T a[3], T b[3])
{
    T accum = 0;
    for (int i=0; i < 3; ++i) {
        accum += (*a++)*(*b++);
    }
    return accum;
}

double cosine_similarity(vec3& p1, vec3& p2) {
    return dot_product_3D(p1.data(), p2.data());
}

template <typename T>
inline T l2_norm_3d(T a[3])
{
    T accum = 0;
    for (int i = 0; i < 3; i++) {
        accum += a[i]*a[i];
    }
    return sqrt(accum);
}

template <typename T>
inline void normalize_vector_3d(T vector[3])
{
    T norm = l2_norm_3d<T>(vector);
    for (int i = 0; i < 3; ++i) {
        vector[i] /= norm;
    }
}

bool _compare(size_t a, size_t b, const std::vector<double>& data)
{
    return data[a]<data[b];
}


std::pair< std::vector<size_t>, std::vector<size_t> > spherical_k_means(const std::vector<vec3>& points, const vec3& center, size_t k, size_t max_iterations, const std::vector<vec3>& centroids_in, bool ensure_k)
{
    if (points.size() < k)
        throw std::runtime_error("Number of points must be larger than number of clusters!");

    auto npoints = points.size();
    std::vector<vec3> normalized_points(npoints);
    for (size_t i=0; i < npoints; ++i)
    {
        auto pt = points[i] - center;
        normalize_vector_3d(pt.data());
        normalized_points[i] = pt;
    }

    std::vector<vec3> centroids(k);

    for (size_t i=0; i<k; ++i)
    {
        auto pt = centroids_in[i] - center;
        normalize_vector_3d(pt.data());
        centroids[i] = pt;
    }

    if (ensure_k)
    {
        std::vector<size_t> closest;
        std::vector<size_t> indices(npoints);
        std::iota(indices.begin(), indices.end(), 0);
        std::vector<double> distances(npoints);

        std::vector<vec3> new_centroids(k);
        for (size_t i=0; i<k; ++i)
        {
            auto& pt = centroids[i];
            for (size_t j=0; j<npoints; ++j)
            {
                distances[j] = -cosine_similarity(pt, normalized_points[j]);
            }
            std::sort(indices.begin(), indices.end(), std::bind(_compare, std::placeholders::_1, std::placeholders::_2, distances));
            for (auto idx: indices) {
                if (std::find(closest.begin(), closest.end(), idx) == closest.end())
                {
                    closest.push_back(idx);
                    new_centroids[i] = normalized_points[idx];
                    break;
                }
            }

        }
        centroids = new_centroids;
    }


    std::vector<size_t> labels(npoints);
    std::vector<size_t> closest;


    // perform k-means clustering
    bool converged=false;
    for (size_t i=0; i<max_iterations; ++i)
    {
        converged=true;
        std::vector<std::vector<vec3>> clusters(k);
        size_t p=0;
        for (auto& point: normalized_points)
        {
            auto best_similarity = -std::numeric_limits<double>::infinity();
            int closest_centroid_index = 0;
            for (size_t j=0; j<k; ++j) {
                auto& centroid = centroids[j];
                auto sim = cosine_similarity(point, centroid);
                if (sim > best_similarity)
                {
                    best_similarity = sim;
                    closest_centroid_index = j;
                }

            }
            if (labels[p] != closest_centroid_index) {
                converged=false;
                labels[p] = closest_centroid_index;
            }
            clusters[closest_centroid_index].push_back(point);
            p++;
        }

        for (size_t j=0; j>k; ++j)
        {
            const auto& cluster = clusters[j];
            if (cluster.size() == 0) {
                // This shouldn't be possible, but just in case...
                continue;
            }
            // In the general case, finding the "surface centroid" of a set of points 
            // randomly scattered on a sphere is decidedly non-trivial - but if we know
            // that the points are limited to a reasonably small area of the surface then
            // we can make a quick, "good enough" approximation by averaging the points 
            // in Cartesian space and re-normalising to r=1.
            double sum_x=0, sum_y=0, sum_z=0;
            for (const auto& point: cluster)
            {
                sum_x += point[0];
                sum_y += point[1];
                sum_z += point[2];
            }
            vec3 new_centroid = {sum_x, sum_y, sum_z};
            normalize_vector_3d(new_centroid.data());   
            centroids[j] = new_centroid;
        }
        if(converged) break;
    }
    for (size_t i=0; i<k; ++i)
    {
        size_t closest_index=0;
        double best_similarity = -std::numeric_limits<double>::infinity();
        auto centroid = centroids[i];
        for (size_t pi=0; pi<labels.size(); ++pi)
        {
            if (labels[pi]==i)
            {
                auto sim = cosine_similarity(normalized_points[pi], centroid);
                if (sim > best_similarity)
                {
                    closest_index = pi;
                    best_similarity = sim;
                }
            }
        }
        closest.push_back(closest_index);
    } 
    return {labels, closest};
}

std::pair< std::vector<size_t>, std::vector<size_t> > spherical_k_means(const std::vector<vec3>& points, const vec3& center, size_t k, size_t max_iterations, unsigned int random_seed)
{
    std::vector<size_t> center_indices;
    std::vector<vec3> centroids(k);
    // initialise k random centroids
    std::default_random_engine e;
    e.seed(random_seed);
    std::uniform_int_distribution<size_t> dist(0, points.size()-1);
    for (size_t i=0; i<k; ++i)
    {
        size_t idx = dist(e);
        while (std::find(center_indices.begin(), center_indices.end(), idx) != center_indices.end()) {
            idx = dist(e);
        }
        centroids[i] = points[idx];
        center_indices.push_back(idx);
    }
    return spherical_k_means(points, center, k, max_iterations, centroids, false);
}

namespace py=pybind11;

PYBIND11_MODULE(_kmeans, m) {
    m.doc() = "k-means clustering of points on a spherical surface. Considers only angular differences "
        "(i.e. all vectors will be normalised to the unit sphere for comparison).";

    m.def("spherical_k_means_random", [](py::array_t<double> points, py::array_t<double> center, size_t k, size_t max_iterations, unsigned int random_seed) {
        if (points.ndim() != 2 || points.shape(1) != 3)
            throw std::runtime_error ("Points should be a n x 3 array of Cartesian coordinates!");
        if (center.ndim() !=1 || center.shape(0) !=3)
            throw std::runtime_error("Center should be a 1 x 3 array giving x,y,z coordinates of the sphere center!");
        std::vector<vec3> pointsvec;
        auto p = points.unchecked<2>();
        for (size_t i=0; i<points.shape(0); ++i)
        {
            pointsvec.push_back(vec3{ {p(i,0), p(i,1), p(i,2)} });
        }
        auto c = center.unchecked<1>();
        vec3 vcenter = {c(0), c(1), c(2)};
        auto result = spherical_k_means(pointsvec, vcenter, k, max_iterations, random_seed);
        py::array rlabels(result.first.size(), result.first.data());
        py::array rclosest(result.second.size(), result.second.data());
        return std::make_tuple(rlabels, rclosest);
    },
        "Perform spherical k-means clustering of the given points, considering only angular distance. All input points "
        "will be mapped to the unit sphere by subtracting `center` and normalising to unit length prior to clustering "
        "into `k` clusters. If `max_iterations` is set to zero the algorithm will run to convergence. The initial "
        "seed centroid for each cluster will be randomly chosen from the input points. The default `random_seed` "
        "is zero. Repeated runs with the same random seed should give identical results.\n"
        "Returns:\n"
        " - an array of integers of the same length as `points`, denoting the cluster to which each point should be assigned;\n"
        " - an array of integers of length `k` giving the index in `points` of the point closest to the centroid of each cluster.",
        py::arg("points"), py::arg("center"), py::arg("k"), py::arg("max_iterations")=0, py::arg("random_seed")=0
    )
    .def("spherical_k_means_defined", [](py::array_t<double> points, py::array_t<double> center, size_t k, py::array_t<double> centroids, size_t max_iterations) {
        if (points.ndim() != 2 || points.shape(1) != 3)
            throw std::runtime_error ("Points should be a n x 3 array of Cartesian coordinates!");
        if (center.ndim() !=1 || center.shape(0) !=3)
            throw std::runtime_error("Center should be a 1 x 3 array giving x,y,z coordinates of the sphere center!");
        if (centroids.ndim() != 2 || centroids.shape(0) != k)
            throw std::runtime_error("Number of seed centroids must equal k!");
        std::vector<vec3> pointsvec(points.shape(0));
        auto p = points.unchecked<2>();
        for (size_t i=0; i<points.shape(0); ++i)
        {
            pointsvec[i] = vec3{ {p(i,0), p(i,1), p(i,2)} };
        }
        auto c = center.unchecked<1>();
        vec3 vcenter = {c(0), c(1), c(2)};
        std::vector<vec3> centroids_vec(centroids.shape(0));
        auto cv = centroids.unchecked<2>();
        for (size_t i=0; i<centroids.shape(0); ++i)
        {
            centroids_vec[i] =vec3{ {cv(i,0), cv(i,1), cv(i,2)}};
        }
        auto result = spherical_k_means(pointsvec, vcenter, k, max_iterations, centroids_vec, true);
        py::array rlabels(result.first.size(), result.first.data());
        py::array rclosest(result.second.size(), result.second.data());
        return std::make_tuple(rlabels, rclosest);
    },
        "Perform spherical k-means clustering of the given points, considering only angular distance. All input points "
        "will be mapped to the unit sphere by subtracting `center` and normalising to unit length prior to clustering "
        "into `k` clusters. If `max_iterations` is set to zero the algorithm will run to convergence. The initial "
        "seed centroid for each cluster will be chosen as the nearest unused point (in angular space) to each point in "
        "`seed_centroids`.\n"
        "Returns:\n"
        " - an array of integers of the same length as `points`, denoting the cluster to which each point should be assigned;\n"
        " - an array of integers of length `k` giving the index in `points` of the point closest to the centroid of each cluster.",
        py::arg("points"), py::arg("center"), py::arg("k"), py::arg("seed_centroids"), py::arg("max_iterations")=0
    );

};

