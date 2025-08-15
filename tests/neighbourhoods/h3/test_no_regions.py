"""Tests for neigbourhoods with no regions."""

import pytest

from srai.neighbourhoods.h3_neighbourhood import H3IndexType, H3Neighbourhood


@pytest.mark.parametrize(  # type: ignore
    "index,expected,expected_with_include_center",
    [
        (
            "811e3ffffffffff",
            {
                "811f3ffffffffff",
                "811fbffffffffff",
                "811ebffffffffff",
                "811efffffffffff",
                "811e7ffffffffff",
                "811f7ffffffffff",
            },
            {
                "811f3ffffffffff",
                "811fbffffffffff",
                "811ebffffffffff",
                "811efffffffffff",
                "811e7ffffffffff",
                "811f7ffffffffff",
                "811e3ffffffffff",
            },
        ),
        (
            "831f0bfffffffff",
            {
                "831f08fffffffff",
                "831f0afffffffff",
                "831e24fffffffff",
                "831e25fffffffff",
                "831f56fffffffff",
                "831f09fffffffff",
            },
            {
                "831f08fffffffff",
                "831f0afffffffff",
                "831e24fffffffff",
                "831e25fffffffff",
                "831f56fffffffff",
                "831f09fffffffff",
                "831f0bfffffffff",
            },
        ),
        (
            "882baa7b69fffff",
            {
                "882baa4e97fffff",
                "882baa7b6dfffff",
                "882baa7b61fffff",
                "882baa7b6bfffff",
                "882baa7b45fffff",
                "882baa4e93fffff",
            },
            {
                "882baa4e97fffff",
                "882baa7b6dfffff",
                "882baa7b61fffff",
                "882baa7b6bfffff",
                "882baa7b45fffff",
                "882baa4e93fffff",
                "882baa7b69fffff",
            },
        ),
        (
            581496515558637567,
            {
                581514107744681983,
                581522903837704191,
                581505311651659775,
                581509709698170879,
                581500913605148671,
                581518505791193087,
            },
            {
                581514107744681983,
                581522903837704191,
                581505311651659775,
                581509709698170879,
                581500913605148671,
                581518505791193087,
                581496515558637567,
            },
        ),
        (
            590517733586632703,
            {
                590517527428202495,
                590517664867155967,
                590501859387506687,
                590501928106983423,
                590522887547387903,
                590517596147679231,
            },
            {
                590517527428202495,
                590517664867155967,
                590501859387506687,
                590501928106983423,
                590522887547387903,
                590517596147679231,
                590517733586632703,
            },
        ),
        (
            613257728762052607,
            {
                613257716730691583,
                613257728766246911,
                613257728753663999,
                613257728764149759,
                613257728724303871,
                613257716726497279,
            },
            {
                613257716730691583,
                613257728766246911,
                613257728753663999,
                613257728764149759,
                613257728724303871,
                613257716726497279,
                613257728762052607,
            },
        ),
    ],
)
def test_get_neighbours(
    index: H3IndexType, expected: set[H3IndexType], expected_with_include_center: set[H3IndexType]
) -> None:
    """Test get_neighbours of H3Neighbourhood."""
    neighbourhood: H3Neighbourhood[H3IndexType] = H3Neighbourhood()
    assert neighbourhood.get_neighbours(index) == expected
    assert neighbourhood.get_neighbours(index, include_center=True) == expected_with_include_center

    neighbourhood_with_include_center: H3Neighbourhood[H3IndexType] = H3Neighbourhood(
        include_center=True
    )
    assert neighbourhood_with_include_center.get_neighbours(index) == expected_with_include_center
    assert neighbourhood_with_include_center.get_neighbours(index, include_center=False) == expected


@pytest.mark.parametrize(  # type: ignore
    "index,distance,expected,expected_with_include_center",
    [
        ("811e3ffffffffff", -2, set(), set()),
        ("811e3ffffffffff", -1, set(), set()),
        ("811e3ffffffffff", 0, set(), {"811e3ffffffffff"}),
        (
            "861f09b27ffffff",
            1,
            {
                "861f09b07ffffff",
                "861f09b2fffffff",
                "861f09b37ffffff",
                "861f7248fffffff",
                "861f7249fffffff",
                "861f724d7ffffff",
            },
            {
                "861f09b07ffffff",
                "861f09b2fffffff",
                "861f09b37ffffff",
                "861f7248fffffff",
                "861f7249fffffff",
                "861f724d7ffffff",
                "861f09b27ffffff",
            },
        ),
        (
            "861f09b27ffffff",
            2,
            {
                "861f0984fffffff",
                "861f0986fffffff",
                "861f09b07ffffff",
                "861f09b0fffffff",
                "861f09b17ffffff",
                "861f09b1fffffff",
                "861f09b2fffffff",
                "861f09b37ffffff",
                "861f09b77ffffff",
                "861f72487ffffff",
                "861f7248fffffff",
                "861f72497ffffff",
                "861f7249fffffff",
                "861f724afffffff",
                "861f724c7ffffff",
                "861f724d7ffffff",
                "861f724dfffffff",
                "861f724f7ffffff",
            },
            {
                "861f0984fffffff",
                "861f0986fffffff",
                "861f09b07ffffff",
                "861f09b0fffffff",
                "861f09b17ffffff",
                "861f09b1fffffff",
                "861f09b2fffffff",
                "861f09b37ffffff",
                "861f09b77ffffff",
                "861f72487ffffff",
                "861f7248fffffff",
                "861f72497ffffff",
                "861f7249fffffff",
                "861f724afffffff",
                "861f724c7ffffff",
                "861f724d7ffffff",
                "861f724dfffffff",
                "861f724f7ffffff",
                "861f09b27ffffff",
            },
        ),
        (581496515558637567, -2, set(), set()),
        (581496515558637567, -1, set(), set()),
        (581496515558637567, 0, set(), {581496515558637567}),
        (
            604028374226042879,
            1,
            {
                604028373689171967,
                604028374360260607,
                604028374494478335,
                604035561451159551,
                604035561719595007,
                604035562659119103,
            },
            {
                604028373689171967,
                604028374360260607,
                604028374494478335,
                604035561451159551,
                604035561719595007,
                604035562659119103,
                604028374226042879,
            },
        ),
        (
            604028374226042879,
            2,
            {
                604028362012229631,
                604028362549100543,
                604028373689171967,
                604028373823389695,
                604028373957607423,
                604028374091825151,
                604028374360260607,
                604028374494478335,
                604028375568220159,
                604035561316941823,
                604035561451159551,
                604035561585377279,
                604035561719595007,
                604035561988030463,
                604035562390683647,
                604035562659119103,
                604035562793336831,
                604035563195990015,
            },
            {
                604028362012229631,
                604028362549100543,
                604028373689171967,
                604028373823389695,
                604028373957607423,
                604028374091825151,
                604028374360260607,
                604028374494478335,
                604028375568220159,
                604035561316941823,
                604035561451159551,
                604035561585377279,
                604035561719595007,
                604035561988030463,
                604035562390683647,
                604035562659119103,
                604035562793336831,
                604035563195990015,
                604028374226042879,
            },
        ),
    ],
)
def test_get_neighbours_up_to_distance(
    index: H3IndexType,
    distance: int,
    expected: set[H3IndexType],
    expected_with_include_center: set[H3IndexType],
) -> None:
    """Test get_neighbours_up_to_distance of H3Neighbourhood."""
    neighbourhood: H3Neighbourhood[H3IndexType] = H3Neighbourhood()
    assert neighbourhood.get_neighbours_up_to_distance(index, distance) == expected
    assert (
        neighbourhood.get_neighbours_up_to_distance(index, distance, include_center=True)
        == expected_with_include_center
    )

    neighbourhood_with_include_center: H3Neighbourhood[H3IndexType] = H3Neighbourhood(
        include_center=True
    )
    assert (
        neighbourhood_with_include_center.get_neighbours_up_to_distance(index, distance)
        == expected_with_include_center
    )
    assert (
        neighbourhood_with_include_center.get_neighbours_up_to_distance(
            index, distance, include_center=False
        )
        == expected
    )


@pytest.mark.parametrize(  # type: ignore
    "index,distance,expected,expected_with_include_center",
    [
        ("811e3ffffffffff", -2, set(), set()),
        ("811e3ffffffffff", -1, set(), set()),
        ("811e3ffffffffff", 0, set(), {"811e3ffffffffff"}),
        (
            "862bac507ffffff",
            1,
            {
                "862bac50fffffff",
                "862bac517ffffff",
                "862bac51fffffff",
                "862bac527ffffff",
                "862bac52fffffff",
                "862bac537ffffff",
            },
            {
                "862bac50fffffff",
                "862bac517ffffff",
                "862bac51fffffff",
                "862bac527ffffff",
                "862bac52fffffff",
                "862bac537ffffff",
            },
        ),
        (
            "862bac507ffffff",
            2,
            {
                "862ba124fffffff",
                "862ba126fffffff",
                "862bac427ffffff",
                "862bac437ffffff",
                "862bac557ffffff",
                "862bac577ffffff",
                "862bac5a7ffffff",
                "862bac5afffffff",
                "862bacc8fffffff",
                "862bacc9fffffff",
                "862baccd7ffffff",
                "862baccdfffffff",
            },
            {
                "862ba124fffffff",
                "862ba126fffffff",
                "862bac427ffffff",
                "862bac437ffffff",
                "862bac557ffffff",
                "862bac577ffffff",
                "862bac5a7ffffff",
                "862bac5afffffff",
                "862bacc8fffffff",
                "862bacc9fffffff",
                "862baccd7ffffff",
                "862baccdfffffff",
            },
        ),
        (581496515558637567, -2, set(), set()),
        (581496515558637567, -1, set(), set()),
        (581496515558637567, 0, set(), {581496515558637567}),
        (
            604250655426609151,
            1,
            {
                604250655560826879,
                604250655695044607,
                604250655829262335,
                604250655963480063,
                604250656097697791,
                604250656231915519,
            },
            {
                604250655560826879,
                604250655695044607,
                604250655829262335,
                604250655963480063,
                604250656097697791,
                604250656231915519,
            },
        ),
        (
            604250655426609151,
            2,
            {
                604249887835422719,
                604249888372293631,
                604250651668512767,
                604250651936948223,
                604250656768786431,
                604250657305657343,
                604250658110963711,
                604250658245181439,
                604250687773081599,
                604250688041517055,
                604250688981041151,
                604250689115258879,
            },
            {
                604249887835422719,
                604249888372293631,
                604250651668512767,
                604250651936948223,
                604250656768786431,
                604250657305657343,
                604250658110963711,
                604250658245181439,
                604250687773081599,
                604250688041517055,
                604250688981041151,
                604250689115258879,
            },
        ),
    ],
)
def test_get_neighbours_at_distance(
    index: H3IndexType,
    distance: int,
    expected: set[H3IndexType],
    expected_with_include_center: set[H3IndexType],
) -> None:
    """Test get_neighbours_at_distance of H3Neighbourhood."""
    neighbourhood: H3Neighbourhood[H3IndexType] = H3Neighbourhood()
    assert neighbourhood.get_neighbours_at_distance(index, distance) == expected
    assert (
        neighbourhood.get_neighbours_at_distance(index, distance, include_center=True)
        == expected_with_include_center
    )

    neighbourhood_with_include_center: H3Neighbourhood[H3IndexType] = H3Neighbourhood(
        include_center=True
    )
    assert (
        neighbourhood_with_include_center.get_neighbours_at_distance(index, distance)
        == expected_with_include_center
    )
    assert (
        neighbourhood_with_include_center.get_neighbours_at_distance(
            index, distance, include_center=False
        )
        == expected
    )
