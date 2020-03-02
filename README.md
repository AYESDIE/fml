# fml

## Fast Machine Learning meets Modern C++.

![](https://cdn.discordapp.com/attachments/478542795452317696/581820786374082580/unknown.png)

**fml** is a Machine Learning Library written in modern C++.


<p>
  <a href="https://discord.gg/FXmq4p2"><img src="https://img.shields.io/badge/discord-join-7289DA.svg?logo=discord&longCache=true&style=flat" /></a>
</p>

### Dependencies

fml has the following dependency:

```
CMake       >= 3.1
```

### Building fml from source:

fml uses CMake as a build system. One can consult any CMake tutorials for further documentation, but this tutorial should be enough to get fml built and installed.

First, unpack the fml source and change into the unpacked directory. Here, we use fml-x.y.z where x.y.z is the version.

```
$ tar -xzf fml-x.y.z.tar.gz
$ cd fml-x.y.z
```

Then, make a build directory. The directory can have any name, not just `build`, but `build` is sufficient.

```
$ mkdir build
$ cd build
```

The next step is to run CMake to configure the project.  Running CMake is the
equivalent to running `./configure` with autotools.

```
$ cmake ../
```

Once CMake is configured, building the library is as simple as typing `make`.

```
$ make
```

If you wish to install fml to `/usr/local/include`, once it has built, make sure you have root privileges (or write permissions to that directory), and simply type

```
# make install
```

### Usage:

```cpp
#include <fml/methods/logistic_regression/logistic_regression.hpp>
#include <fml/core/optimizers/sgd/sgd.hpp>
#include <fml/core/math/normalize.hpp>
#include <fml/core.hpp>

using namespace fml;
using namespace fml::regression;

int main()
{
    std::ifstream in_file;
    in_file.open("data/logistictest.csv");
    auto dataset = xt::load_csv<double>(in_file);
    in_file.close();
    
    xt::xtensor<size_t, 2> labels = xt::view(dataset, xt::all(), xt::keep(3));
    xt::xtensor<double, 2> data = xt::view(dataset, xt::all(), xt::keep(0, 1, 2));
    
    fml::math::Normalize(data, 0, 1, 2);
    
    fml::optimizer::SGD sgd(0.1, 100000, 1e-9, 20);
    LogisticRegression<> lr(data, labels, sgd);
    
    xt::xtensor<size_t, 2> pred;
    lr.Compute(data, pred);

    return 0;
}
```