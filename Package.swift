// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "KaitoAccelLab",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "AccelLab", targets: ["AccelLab"]),
        .executable(name: "accel-lab", targets: ["accel-lab"]),
    ],
    dependencies: [
        .package(path: "../KaitoKit"),
        .package(path: "../GyoshukuKit"),
    ],
    targets: [
        .target(name: "AccelLab", dependencies: ["KaitoKit", "GyoshukuKit"],
                linkerSettings: [.linkedFramework("Metal"), .linkedFramework("MetalPerformanceShaders"),
                                 .linkedFramework("MetalPerformanceShadersGraph"), .linkedFramework("CoreML"),
                                 .linkedFramework("Accelerate"), .linkedLibrary("compression"),
                                 .linkedLibrary("z"), .linkedLibrary("bz2")]),
        .executableTarget(name: "accel-lab", dependencies: ["AccelLab"]),
        .testTarget(name: "AccelLabTests", dependencies: ["AccelLab"]),
    ]
)
