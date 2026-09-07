pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Window
import "pages"

/*!
    Main - root of the control window (docs/GLASS_DESIGN.md §1.1, §2.1).

    root = window.  The visible slab is inset 36 / 28 / 36 / 52 logical px
    (default window 832x640 -> slab 760x560, minimum 752x560 -> 680x480); the
    margin holds the slab shadow and nothing else is painted there.
    Layers, top to bottom: backdrop image (hidden source of the sharp texture),
    the 1/4-resolution blur chain (every hop hidden), the slab shadow, the
    "regular" glass slab and the content layer (TitleBar, GlassSegmentedBar,
    the four pages, StatusStrip) inset by Theme.pad.

    Everything interactive is laid out and hit-tested on the slab rect: the
    title bar drag and the 8 px resize edges call bridge.startMove() /
    bridge.startResize(edges) from inside the press.  Keyboard: Ctrl+1..4 and
    Ctrl+Tab / Ctrl+Shift+Tab switch tabs, Esc closes an open popup (handled by
    the popups), Alt+Space opens the window menu.  Theme.pointer follows the
    HoverHandler directly while hovered and glides to Theme.pointerRest in
    240 ms when the pointer leaves.
*/
// qmllint disable unqualified
Item {
    id: root

    width: 832
    height: 640
    focus: true

    readonly property real dpr: Screen.devicePixelRatio > 0 ? Screen.devicePixelRatio : 1
    readonly property bool backdropReady: bridge.backdropSerial > 0 && backdrop.status === Image.Ready
    // exit position of the pointer highlight; glides to Theme.pointerRest after hover ends
    property real exitX: -2000
    property real exitY: -2000

    Component.onCompleted: {
        Theme.rootSize = Qt.binding(function () { return Qt.size(root.width, root.height) })
        Theme.pointer = Qt.binding(function () {
            return hover.hovered ? hover.point.position : Qt.point(root.exitX, root.exitY)
        })
    }

    HoverHandler {
        id: hover
        onHoveredChanged: {
            if (hovered) {
                exitAnim.stop()
            } else {
                root.exitX = point.position.x
                root.exitY = point.position.y
                exitAnim.restart()
            }
        }
    }
    ParallelAnimation {
        id: exitAnim
        NumberAnimation { target: root; property: "exitX"; to: Theme.pointerRest.x; duration: 240; easing.type: Easing.OutCubic }
        NumberAnimation { target: root; property: "exitY"; to: Theme.pointerRest.y; duration: 240; easing.type: Easing.OutCubic }
    }

    // ------------------------------------------------------------ backdrop + sources
    // Sized from bridge.backdropSize (logical = physical / dpr): Qt Quick ignores the
    // devicePixelRatio of image-provider images, so the Image's implicit size is the
    // physical texture size and would magnify the backdrop at 125 % / 150 % scaling.
    Item {
        id: backdropLayer
        x: bridge.backdropOrigin.x
        y: bridge.backdropOrigin.y
        width: bridge.backdropSize.width > 8 ? bridge.backdropSize.width : root.width + 64
        height: bridge.backdropSize.height > 8 ? bridge.backdropSize.height : root.height + 64
        Image {
            id: backdrop
            anchors.fill: parent
            source: bridge.backdropSerial > 0 ? "image://backdrop/" + bridge.backdropSerial : ""
            cache: false
            smooth: true
            asynchronous: false
            fillMode: Image.Stretch
        }
    }
    ShaderEffectSource {
        id: sharpSrc
        sourceItem: backdropLayer
        hideSource: true
        live: true
        smooth: true
        textureSize: Qt.size(backdropLayer.width * root.dpr, backdropLayer.height * root.dpr)
    }
    // 1/4-resolution blur chain H -> V -> H -> V (sigma ~ 10 logical px).  The effects
    // stay visible so they render into their layers; every source hides its item so
    // nothing of the chain is composited on screen.
    ShaderEffect {
        id: blurH1
        x: backdropLayer.x; y: backdropLayer.y
        width: backdropLayer.width / 4; height: backdropLayer.height / 4
        property var source: sharpSrc
        property point step: Qt.point(1 / (width * root.dpr), 0)
        fragmentShader: "shaders/blur.frag.qsb"
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
        fragmentShader: "shaders/blur.frag.qsb"
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
        fragmentShader: "shaders/blur.frag.qsb"
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
        fragmentShader: "shaders/blur.frag.qsb"
    }
    ShaderEffectSource {
        id: frostSrc; sourceItem: blurV2; hideSource: true; live: true; smooth: true; mipmap: false
        textureSize: Qt.size(blurV2.width * root.dpr, blurV2.height * root.dpr)
    }

    // ------------------------------------------------------------ slab
    GlassShadow {
        id: slabShadow
        x: slab.x; y: slab.y; width: slab.width; height: slab.height
        radius: slab.radius
        sigma: 12
        offsetY: 8
    }
    GlassSurface {
        id: slab
        objectName: "slab"
        variant: "regular"
        x: 36
        y: 28
        width: root.width - 72
        height: root.height - 80
        radius: Theme.radiusWindow
        sourceItem: backdropLayer
        sharpSource: sharpSrc
        frostSource: frostSrc
        // until the first backdrop frame the glass path would sample an empty texture
        solidMode: Theme.mode === "glass" && root.backdropReady ? 0 : 1
    }

    // ------------------------------------------------------------ content
    Item {
        id: content
        objectName: "content"
        x: slab.x + Theme.pad
        y: slab.y + Theme.pad
        width: slab.width - 2 * Theme.pad
        height: slab.height - 2 * Theme.pad

        TitleBar {
            id: titleBar
            objectName: "titleBar"
            anchors.top: parent.top
            anchors.left: parent.left
            anchors.right: parent.right
            title: bridge.windowTitle
            running: bridge.running
            onDragRequested: bridge.startMove()
            onMinimizeRequested: bridge.minimize()
            onCloseRequested: bridge.close()
        }

        GlassSegmentedBar {
            id: tabBar
            objectName: "tabBar"
            anchors.top: titleBar.bottom
            anchors.topMargin: Theme.gap
            anchors.left: parent.left
            anchors.right: parent.right
        }

        // page stack: 160 ms opacity crossfade + 10 px slide (instant under reduceMotion)
        Item {
            id: pages
            objectName: "pages"
            anchors.top: tabBar.bottom
            anchors.topMargin: Theme.gap
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.bottom: statusStrip.top
            anchors.bottomMargin: Theme.gap

            component PageSlot: Item {
                id: slot
                required property int index
                readonly property bool active: tabBar.currentIndex === index
                anchors.fill: parent
                opacity: active ? 1 : 0
                x: active ? 0 : (index < tabBar.currentIndex ? -10 : 10)
                visible: opacity > 0.001
                enabled: active
                Behavior on opacity { enabled: !Theme.reduceMotion; NumberAnimation { duration: Theme.fade; easing.type: Easing.OutCubic } }
                Behavior on x { enabled: !Theme.reduceMotion; NumberAnimation { duration: Theme.fade; easing.type: Easing.OutCubic } }
            }
            PageSlot { index: 0; TranslatePage { objectName: "translatePage"; anchors.fill: parent } }
            PageSlot { index: 1; OverlayPage { objectName: "overlayPage"; anchors.fill: parent } }
            PageSlot { index: 2; EnginesPage { objectName: "enginesPage"; anchors.fill: parent } }
            PageSlot { index: 3; HotkeysPage { objectName: "hotkeysPage"; anchors.fill: parent } }
        }

        StatusStrip {
            id: statusStrip
            objectName: "statusStrip"
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.bottom: parent.bottom
            message: bridge.statusMessage
            running: bridge.running
            totalText: bridge.stats.totalText
            fpsText: bridge.stats.fpsText
            stagesText: bridge.stats.stagesText
            segmentsText: bridge.stats.segmentsText
            cacheText: bridge.stats.cacheText
            devicesText: bridge.stats.devicesText
        }
    }

    // ------------------------------------------------------------ resize edges (8 px, on the slab rect)
    component ResizeEdge: MouseArea {
        property int edges: 0
        acceptedButtons: Qt.LeftButton
        onPressed: bridge.startResize(edges)
    }
    ResizeEdge { x: slab.x; y: slab.y + 8; width: 8; height: slab.height - 16; edges: Qt.LeftEdge; cursorShape: Qt.SizeHorCursor }
    ResizeEdge { x: slab.x + slab.width - 8; y: slab.y + 8; width: 8; height: slab.height - 16; edges: Qt.RightEdge; cursorShape: Qt.SizeHorCursor }
    ResizeEdge { x: slab.x + 8; y: slab.y; width: slab.width - 16; height: 8; edges: Qt.TopEdge; cursorShape: Qt.SizeVerCursor }
    ResizeEdge { x: slab.x + 8; y: slab.y + slab.height - 8; width: slab.width - 16; height: 8; edges: Qt.BottomEdge; cursorShape: Qt.SizeVerCursor }
    ResizeEdge { x: slab.x; y: slab.y; width: 8; height: 8; edges: Qt.LeftEdge | Qt.TopEdge; cursorShape: Qt.SizeFDiagCursor }
    ResizeEdge { x: slab.x + slab.width - 8; y: slab.y + slab.height - 8; width: 8; height: 8; edges: Qt.RightEdge | Qt.BottomEdge; cursorShape: Qt.SizeFDiagCursor }
    ResizeEdge { x: slab.x + slab.width - 8; y: slab.y; width: 8; height: 8; edges: Qt.RightEdge | Qt.TopEdge; cursorShape: Qt.SizeBDiagCursor }
    ResizeEdge { x: slab.x; y: slab.y + slab.height - 8; width: 8; height: 8; edges: Qt.LeftEdge | Qt.BottomEdge; cursorShape: Qt.SizeBDiagCursor }

    // ------------------------------------------------------------ keyboard
    function selectPage(index) {
        var n = tabBar.count
        tabBar.currentIndex = (index + n) % n
    }
    function openWindowMenu() {
        windowMenu.x = content.x
        windowMenu.y = content.y + titleBar.height
        windowMenu.open()
    }
    Shortcut { sequence: "Ctrl+1"; onActivated: root.selectPage(0) }
    Shortcut { sequence: "Ctrl+2"; onActivated: root.selectPage(1) }
    Shortcut { sequence: "Ctrl+3"; onActivated: root.selectPage(2) }
    Shortcut { sequence: "Ctrl+4"; onActivated: root.selectPage(3) }
    Shortcut { sequence: "Ctrl+Tab"; onActivated: root.selectPage(tabBar.currentIndex + 1) }
    Shortcut { sequences: ["Ctrl+Shift+Tab", "Ctrl+Shift+Backtab"]; onActivated: root.selectPage(tabBar.currentIndex - 1) }
    Shortcut { sequence: "Alt+Space"; onActivated: root.openWindowMenu() }

    WindowMenu {
        id: windowMenu
        objectName: "windowMenu"
        onMinimizeRequested: bridge.minimize()
        onMoveRequested: bridge.keyboardMove()
        onSizeRequested: bridge.keyboardSize()
        onCloseRequested: bridge.close()
    }
}
