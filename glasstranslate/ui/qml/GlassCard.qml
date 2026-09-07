pragma ComponentBehavior: Bound
import QtQuick

/*!
    GlassCard - a group inset (docs/GLASS_DESIGN.md §2.1, §2.3): the fill path
    with `Theme.fillCard`, radius `Theme.radiusCard`, padding 12, no shadow.
    Children (normally FormRows) go into a Column below the optional `title`.
*/
Item {
    id: card

    property string title: ""
    property real padding: 12
    property real spacing: Theme.gap
    default property alias content: column.data

    width: parent ? parent.width : 320
    implicitWidth: 320
    implicitHeight: column.implicitHeight + 2 * padding

    Accessible.role: Accessible.Grouping
    Accessible.name: title

    GlassSurface {
        anchors.fill: parent
        variant: "card"
        radius: Theme.radiusCard
    }

    Column {
        id: column
        x: card.padding
        y: card.padding
        width: card.width - 2 * card.padding
        spacing: card.spacing

        Text {
            width: parent.width
            visible: card.title.length > 0
            text: card.title
            font.family: Theme.family
            font.pointSize: Theme.fontBody
            font.weight: 600
            color: Theme.ink
            elide: Text.ElideRight
        }
    }
}
