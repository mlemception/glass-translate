pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Window
import "../../glasstranslate/ui/qml"

/*
    Lab scene for demo/glass_lab.py.  Mirrors the Main.qml layer stack of
    docs/GLASS_DESIGN.md §2.1 over a static backdrop image: backdrop layer,
    sharp source, 1/4-res blur chain, slab shadow, regular slab, content.
    Everything the harness drives comes from the `lab` context object.
*/
// qmllint disable unqualified
Item {
    id: root
    width: 832
    height: 640

    readonly property real dpr: Screen.devicePixelRatio > 0 ? Screen.devicePixelRatio : 1

    // facts the harness reads back
    readonly property string themeMode: Theme.mode
    readonly property real themeInk: Theme.inkValue
    readonly property bool themeHasContext: Theme.hasAppearance && Theme.hasBridge
    readonly property color themeFill: Theme.fill
    readonly property color themeBorder: Theme.borderColor
    readonly property bool backdropReady: backdrop.status === Image.Ready
    readonly property real slabStrengthEffective: slab.effectiveStrength
    readonly property rect slabRect: Qt.rect(slab.x, slab.y, slab.width, slab.height)
    readonly property real slabRadius: slab.radius
    readonly property real trackRadius: track.radius
    readonly property real indicatorRadius: indicator.radius
    readonly property rect trackRect: Qt.rect(trackLayer.x + slabContent.x + slab.x, trackLayer.y + slabContent.y + slab.y, trackLayer.width, trackLayer.height)

    // Main.qml does the same: root size and pointer into the Theme singleton
    // (direct binding, no Behavior - a smoothed light source lags).
    Component.onCompleted: {
        Theme.rootSize = Qt.binding(function () { return Qt.size(root.width, root.height) })
        Theme.pointer = Qt.binding(function () {
            return lab.pointerDriven ? Qt.point(lab.pointerX, lab.pointerY)
                                     : (hover.hovered ? hover.point.position : Theme.pointerRest)
        })
    }
    HoverHandler { id: hover }

    // ------------------------------------------------------------ backdrop + sources
    Item {
        id: backdropLayer
        x: -32
        y: -32
        width: backdrop.implicitWidth > 0 ? backdrop.implicitWidth : root.width + 64
        height: backdrop.implicitHeight > 0 ? backdrop.implicitHeight : root.height + 64
        Image {
            id: backdrop
            anchors.fill: parent
            source: lab.backdropSource
            cache: false
            smooth: true
            fillMode: Image.Stretch
        }
    }
    ShaderEffectSource {
        id: sharpSrc
        sourceItem: backdropLayer
        hideSource: !lab.showBackdrop
        live: true
        smooth: true
        textureSize: Qt.size(backdropLayer.width * root.dpr, backdropLayer.height * root.dpr)
    }
    // 1/4-resolution blur chain H -> V -> H -> V (sigma ~ 10 logical px).  The
    // effects stay visible so they render into their layers; every source hides
    // them so nothing of the chain is composited on screen.
    ShaderEffect {
        id: blurH1
        x: backdropLayer.x; y: backdropLayer.y
        width: backdropLayer.width / 4; height: backdropLayer.height / 4
        property var source: sharpSrc
        property point step: Qt.point(1 / (width * root.dpr), 0)
        fragmentShader: "../../glasstranslate/ui/qml/shaders/blur.frag.qsb"
    }
    ShaderEffectSource {
        id: srcH1; sourceItem: blurH1; hideSource: true; live: true; smooth: true; mipmap: false
        textureSize: Qt.size(blurH1.width * root.dpr, blurH1.height * root.dpr)
    }
    ShaderEffect {
        id: blurV1
        x: backdropLayer.x; y: backdropLayer.y
        width: blurH1.width; height: blurH1.height
        property var source: srcH1
        property point step: Qt.point(0, 1 / (height * root.dpr))
        fragmentShader: "../../glasstranslate/ui/qml/shaders/blur.frag.qsb"
    }
    ShaderEffectSource {
        id: srcV1; sourceItem: blurV1; hideSource: true; live: true; smooth: true; mipmap: false
        textureSize: Qt.size(blurV1.width * root.dpr, blurV1.height * root.dpr)
    }
    ShaderEffect {
        id: blurH2
        x: backdropLayer.x; y: backdropLayer.y
        width: blurH1.width; height: blurH1.height
        property var source: srcV1
        property point step: Qt.point(1 / (width * root.dpr), 0)
        fragmentShader: "../../glasstranslate/ui/qml/shaders/blur.frag.qsb"
    }
    ShaderEffectSource {
        id: srcH2; sourceItem: blurH2; hideSource: true; live: true; smooth: true; mipmap: false
        textureSize: Qt.size(blurH2.width * root.dpr, blurH2.height * root.dpr)
    }
    ShaderEffect {
        id: blurV2
        x: backdropLayer.x; y: backdropLayer.y
        width: blurH1.width; height: blurH1.height
        property var source: srcH2
        property point step: Qt.point(0, 1 / (height * root.dpr))
        fragmentShader: "../../glasstranslate/ui/qml/shaders/blur.frag.qsb"
    }
    ShaderEffectSource {
        id: frostSrc; sourceItem: blurV2; hideSource: true; live: true; smooth: true; mipmap: false
        textureSize: Qt.size(blurV2.width * root.dpr, blurV2.height * root.dpr)
    }

    // ------------------------------------------------------------ slab
    GlassShadow {
        x: slab.x; y: slab.y; width: slab.width; height: slab.height
        radius: slab.radius
        sigma: 12
        offsetY: 8
        visible: lab.showSlab && lab.showShadow && Theme.mode !== "hc"
    }
    GlassSurface {
        id: slab
        variant: "regular"
        x: 36
        y: 28
        width: root.width - 72
        height: root.height - 80
        visible: lab.showSlab
        sourceItem: backdropLayer
        sharpSource: sharpSrc
        frostSource: frostSrc
        frost: lab.rawSlab ? 0 : 1
        frostLift: lab.rawSlab ? 0 : 0.30
        coolLean: lab.rawSlab ? 0 : 0.5
        shadowInner: lab.rawSlab ? 0 : 0.08
        strength: lab.slabStrength >= 0 ? lab.slabStrength : 0.38
        specular: lab.slabSpecular >= 0 ? lab.slabSpecular : 1
    }

    // ------------------------------------------------------------ content (inside the slab, margins 14)
    Item {
        id: slabContent
        x: slab.x + Theme.pad
        y: slab.y + Theme.pad
        width: slab.width - 2 * Theme.pad
        height: slab.height - 2 * Theme.pad
        visible: lab.showSlab && lab.showContent

        Text {
            x: 0; y: 6
            text: "GlassTranslate"
            font.family: Theme.family
            font.pointSize: Theme.fontTitle
            font.weight: 600
            color: Theme.ink
        }
        Text {
            anchors.right: parent.right; y: 10
            text: lab.caption.length > 0 ? lab.caption : ("mode " + Theme.mode + "  ink " + Theme.inkValue.toFixed(2) + "  dpr " + root.dpr.toFixed(2))
            font.family: Theme.mono
            font.pointSize: Theme.fontCaption
            color: Theme.inkSecondary
        }

        // capsule track + pill indicator lensing the track (segmented bar)
        Item {
            id: trackLayer
            x: 0; y: 48
            width: 440; height: 36
            GlassSurface {
                id: track
                anchors.fill: parent
                variant: "card"
                fillColor: Theme.fillTrack
                radius: height / 2
                squircle: 2
            }
            Row {
                anchors.fill: parent
                Repeater {
                    model: ["Translate", "Overlay", "Engines", "Hotkeys"]
                    Item {
                        id: segment
                        required property int index
                        required property string modelData
                        width: trackLayer.width / 4; height: trackLayer.height
                        Text {
                            anchors.centerIn: parent
                            text: segment.modelData
                            font.family: Theme.family
                            font.pointSize: Theme.fontBody
                            font.weight: segment.index === lab.indicatorIndex ? 600 : 400
                            color: segment.index === lab.indicatorIndex ? Theme.ink : Theme.inkSecondary
                            // in solid / HC the pill is opaque: the selected label is drawn above it instead
                            visible: segment.index !== lab.indicatorIndex || Theme.mode === "glass"
                        }
                    }
                }
            }
        }
        ShaderEffectSource {
            id: trackSrc
            sourceItem: trackLayer
            hideSource: false
            live: true
            smooth: true
            textureSize: Qt.size(trackLayer.width * root.dpr, trackLayer.height * root.dpr)
        }
        GlassShadow {
            x: indicator.x; y: indicator.y; width: indicator.width; height: indicator.height
            radius: indicator.radius; sigma: 5; offsetY: 2; color: Qt.rgba(0, 0, 0, 0.18)
        }
        GlassSurface {
            id: indicator
            variant: "pill"
            readonly property real inset: 3
            x: trackLayer.x + inset + lab.indicatorIndex * (trackLayer.width / 4)
            y: trackLayer.y + inset
            width: trackLayer.width / 4 - 2 * inset
            height: trackLayer.height - 2 * inset
            radius: Theme.concentric(track.radius, inset)
            sourceItem: trackLayer
            sharpSource: trackSrc
            Behavior on x {
                enabled: !Theme.reduceMotion
                SpringAnimation { spring: Theme.springLead.spring; damping: Theme.springLead.damping }
            }
        }
        // The selected label is drawn ABOVE the indicator in solid / HC, where the
        // pill is opaque and hides the copy in the lensed layer beneath it.
        Text {
            visible: Theme.mode !== "glass"
            x: indicator.x + (indicator.width - width) / 2
            y: indicator.y + (indicator.height - height) / 2
            text: ["Translate", "Overlay", "Engines", "Hotkeys"][lab.indicatorIndex]
            font.family: Theme.family
            font.pointSize: Theme.fontBody
            font.weight: 600
            color: Theme.ink
        }

        // slider-like track with a pill thumb lensing track + fill
        Item {
            id: sliderLayer
            x: 0; y: 104
            width: 440; height: 24
            Rectangle {
                id: sliderTrack
                anchors.verticalCenter: parent.verticalCenter
                x: 10; width: parent.width - 20; height: 4; radius: 2
                color: Theme.fillTrack
            }
            Rectangle {
                anchors.verticalCenter: parent.verticalCenter
                x: sliderTrack.x; width: sliderTrack.width * lab.sliderValue; height: 4; radius: 2
                color: Theme.accent
            }
        }
        ShaderEffectSource {
            id: sliderSrc
            sourceItem: sliderLayer
            hideSource: false
            live: true
            smooth: true
            textureSize: Qt.size(sliderLayer.width * root.dpr, sliderLayer.height * root.dpr)
        }
        GlassShadow {
            x: thumb.x; y: thumb.y; width: thumb.width; height: thumb.height
            radius: thumb.radius; sigma: 5; offsetY: 2; color: Qt.rgba(0, 0, 0, 0.18)
        }
        GlassSurface {
            id: thumb
            variant: "pill"
            width: 20; height: 20
            x: sliderLayer.x + sliderTrack.x + sliderTrack.width * lab.sliderValue - width / 2
            y: sliderLayer.y + (sliderLayer.height - height) / 2
            radius: height / 2
            sourceItem: sliderLayer
            sharpSource: sliderSrc
        }

        // three fill buttons
        Row {
            id: buttons
            x: 0; y: 146
            spacing: Theme.gap
            Repeater {
                model: ["Start", "Grab mode", "Show glass"]
                Item {
                    id: button
                    required property int index
                    required property string modelData
                    width: 130; height: Theme.controlH
                    GlassShadow { anchors.fill: parent; radius: Theme.radiusControl; sigma: 5; offsetY: 2; color: Qt.rgba(0, 0, 0, 0.16) }
                    GlassSurface {
                        anchors.fill: parent
                        variant: "fill"
                        radius: Theme.radiusControl
                        fillColor: button.index === 0 ? Theme.fillPrimary : Theme.fillControl
                    }
                    Text {
                        anchors.centerIn: parent
                        text: button.modelData
                        font.family: Theme.family
                        font.pointSize: Theme.fontBody
                        font.weight: button.index === 0 ? 600 : 500
                        color: Theme.ink
                    }
                }
            }
        }

        // one card
        GlassSurface {
            id: card
            x: 0; y: 200
            width: 440; height: 150
            variant: "card"
            radius: Theme.radiusCard
        }
        Column {
            x: card.x + 12; y: card.y + 12
            spacing: 6
            Text { text: "Languages"; font.family: Theme.family; font.pointSize: Theme.fontBody; font.weight: 600; color: Theme.ink }
            Text { text: "Source   Auto-detect"; font.family: Theme.family; font.pointSize: Theme.fontBody; color: Theme.ink }
            Text { text: "Target   English"; font.family: Theme.family; font.pointSize: Theme.fontBody; color: Theme.ink }
            Text { text: "Detected language is refreshed on every pass."; font.family: Theme.family; font.pointSize: Theme.fontCaption; color: Theme.inkSecondary }
        }

        // status strip (fill, concentric with the slab)
        Item {
            x: 0; y: parent.height - Theme.stripH
            width: parent.width; height: Theme.stripH
            GlassShadow { anchors.fill: parent; radius: Theme.concentric(Theme.radiusWindow, Theme.pad); sigma: 5; offsetY: 2; color: Qt.rgba(0, 0, 0, 0.16) }
            GlassSurface { anchors.fill: parent; variant: "fill"; radius: Theme.concentric(Theme.radiusWindow, Theme.pad) }
            Text {
                anchors.verticalCenter: parent.verticalCenter; x: 14
                text: "Ready"; font.family: Theme.family; font.pointSize: Theme.fontBody; color: Theme.ink
            }
            Text {
                anchors.verticalCenter: parent.verticalCenter; anchors.right: parent.right; anchors.rightMargin: 14
                text: "Total 12.3 ms · 8.1 fps"; font.family: Theme.mono; font.pointSize: Theme.fontBody; color: Theme.ink
            }
        }
    }
}
