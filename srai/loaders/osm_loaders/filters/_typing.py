"""Module contains a dedicated type alias for OSM tags filter."""

from typing import Union, cast

from srai._typing import is_expected_type

OsmTagsFilter = dict[str, Union[list[str], str, bool]]

GroupedOsmTagsFilter = dict[str, OsmTagsFilter]


def merge_osm_tags_filter(tags: Union[OsmTagsFilter, GroupedOsmTagsFilter]) -> OsmTagsFilter:
    """
    Merge OSM tags filter into `osm_tags_type` type.

    Optionally merges `GroupedOsmTagsFilter` into `OsmTagsFilter` to allow loaders to load all
    defined groups during single operation.

    Args:
        tags (Union[OsmTagsFilter, GroupedOsmTagsFilter]): OSM tags filter definition.

    Raises:
        AttributeError: When provided tags don't match both
            `OsmTagsFilter` or `GroupedOsmTagsFilter`.

    Returns:
        osm_tags_type: Merged filters.
    """
    if is_expected_type(tags, OsmTagsFilter):
        return cast(OsmTagsFilter, tags)
    elif is_expected_type(tags, GroupedOsmTagsFilter):
        return merge_grouped_osm_tags_filter(cast(GroupedOsmTagsFilter, tags))

    raise AttributeError(
        "Provided tags don't match required type definitions"
        " (OsmTagsFilter or GroupedOsmTagsFilter)."
    )


def merge_grouped_osm_tags_filter(grouped_filter: GroupedOsmTagsFilter) -> OsmTagsFilter:
    """
    Merge grouped osm tags filter into a base one.

    Function merges all filter categories into a single one for an OSM loader to use.

    Args:
        grouped_filter (GroupedOsmTagsFilter): Grouped filter to be merged into a single one.

    Returns:
        osm_tags_type: Merged filter.
    """
    if not is_expected_type(grouped_filter, GroupedOsmTagsFilter):
        raise ValueError(
            "Provided filter doesn't match required `GroupedOsmTagsFilter` definition."
        )

    result: OsmTagsFilter = {}
    for sub_filter in grouped_filter.values():
        for osm_tag_key, osm_tag_value in sub_filter.items():
            if osm_tag_key not in result:
                result[osm_tag_key] = []

            # If filter is already a positive boolean, skip
            if isinstance(result[osm_tag_key], bool) and result[osm_tag_key]:
                continue

            # Check bool
            if isinstance(osm_tag_value, bool) and osm_tag_value:
                result[osm_tag_key] = True
            # Check string
            elif isinstance(osm_tag_value, str) and osm_tag_value not in cast(
                list[str], result[osm_tag_key]
            ):
                cast(list[str], result[osm_tag_key]).append(osm_tag_value)
            # Check list
            elif isinstance(osm_tag_value, list):
                new_values = [
                    value
                    for value in osm_tag_value
                    if value not in cast(list[str], result[osm_tag_key])
                ]
                cast(list[str], result[osm_tag_key]).extend(new_values)

    return result
