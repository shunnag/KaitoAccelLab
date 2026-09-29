struct BlockDesc {
    let srcOffset: UInt32
    let srcLength: UInt32
    let dstOffset: UInt32
    let dstLength: UInt32
    let isStored: UInt32
    var pad: (UInt32, UInt32, UInt32) = (0, 0, 0)
}
