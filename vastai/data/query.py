# Implements an object oriented query language for Vast standard query (vast/web/apiquery.py)

from typing import Any, Optional


class Query:
    def __init__(self, query_dict: dict[str, dict[str, Any]]):
        # self.query has format {column : {op : value}}; may have multiple ops per column and multiple columns.
        self.query = query_dict

    @classmethod
    def search_defaults(
        cls,
        verified: Optional[bool] = True,
        rentable: Optional[bool] = True,
        rented: Optional[bool] = False,
    ) -> "Query":
        """Creates a Query pre-populated with default search filters for end-user use."""
        q = cls({})
        for name, value in [("verified", verified), ("rentable", rentable), ("rented", rented)]:
            if value is not None:
                q.query[name] = {"eq": value}
        return q

    def unparse_query(self) -> str:
        """Converts this Query back into a query string that parse_query() would parse into the same dict.

        Inverse of vast/web/parse.py:parse_query, i.e. parse_query(query.unparse_query()) == query.query.
        """
        op_to_str = {
            "eq": "=",
            "neq": "!=",
            "gt": ">",
            "gte": ">=",
            "lt": "<",
            "lte": "<=",
            "in": " in ",
            "notin": " notin ",
        }
        # Fields whose values get multiplied by these factors during parsing
        mult_fields = {
            "cpu_ram": 1000,
            "gpu_ram": 1000,
            "gpu_total_ram": 1000,
            "duration": 24.0 * 60.0 * 60.0,
        }

        parts = []
        for field, ops in self.query.items():
            for op, value in ops.items():
                op_str = op_to_str[op]

                # Reverse the multiplier applied during parsing
                if field in mult_fields:
                    value = value / mult_fields[field]
                    # Use int if it's a whole number
                    if value == int(value):
                        value = int(value)

                # Format the value
                if isinstance(value, bool):
                    val_str = "true" if value else "false"
                elif value is None:
                    val_str = "None"
                elif isinstance(value, list):
                    # in/notin: reverse the space→underscore replacement per element
                    items = [str(v).replace(" ", "_") for v in value]
                    val_str = ",".join(items)
                elif isinstance(value, str):
                    # Reverse the underscore→space replacement done by parse_query
                    val_str = value.replace(" ", "_")
                else:
                    val_str = str(value)

                parts.append(f"{field}{op_str}{val_str}")
        return " ".join(parts)

    def extend(self, other: "Query") -> "Query":
        """
        Extends self.query with filters in other.

        self.query should now have all columns that exist in either self.query or other.query;
        and each column in self.query should have all corresponding ops in either query.
        """
        for column, ops in other.query.items():
            if column in self.query:
                conflicts = self.query[column].keys() & ops.keys()
                if conflicts:
                    raise ValueError(
                        f"Cannot redefine op(s) {conflicts} for column '{column}'"
                    )
                self.query[column].update(ops)
            else:
                self.query[column] = dict(ops)
        return self


class Column:
    def __init__(self, name: str):
        self.name = name

    def __eq__(self, value: Any) -> Query:
        return Query({self.name: {"eq": value}})

    def __ne__(self, value: Any) -> Query:
        return Query({self.name: {"neq": value}})

    def __lt__(self, value: Any) -> Query:
        return Query({self.name: {"lt": value}})

    def __le__(self, value: Any) -> Query:
        return Query({self.name: {"lte": value}})

    def __gt__(self, value: Any) -> Query:
        return Query({self.name: {"gt": value}})

    def __ge__(self, value: Any) -> Query:
        return Query({self.name: {"gte": value}})

    def in_(self, values) -> Query:
        return Query({self.name: {"in": values}})

    def notin_(self, values) -> Query:
        return Query({self.name: {"notin": values}})


# Pre-initialized Column objects for all columns in ask_contract_offers
id_                  = Column("id")
ask_contract_id      = Column("ask_contract_id")
bundle_id            = Column("bundle_id")
bundled_results      = Column("bundled_results")
bw_nvlink            = Column("bw_nvlink")
compute_cap          = Column("compute_cap")
cpu_arch             = Column("cpu_arch")
cpu_cores            = Column("cpu_cores")
cpu_cores_effective  = Column("cpu_cores_effective")
cpu_ghz              = Column("cpu_ghz")
cpu_name             = Column("cpu_name")
cpu_ram              = Column("cpu_ram")
credit_discount_max  = Column("credit_discount_max")
cuda_max_good        = Column("cuda_max_good")
direct_port_count    = Column("direct_port_count")
disk_bw              = Column("disk_bw")
disk_name            = Column("disk_name")
disk_space           = Column("disk_space")
dlperf               = Column("dlperf")
dlperf_per_dphtotal  = Column("dlperf_per_dphtotal")
dph_base             = Column("dph_base")
dph_total            = Column("dph_total")
driver_version       = Column("driver_version")
driver_vers          = Column("driver_vers")
duration             = Column("duration")
end_date             = Column("end_date")
external             = Column("external")
flops_per_dphtotal   = Column("flops_per_dphtotal")
geolocation          = Column("geolocation")
geolocode            = Column("geolocode")
gpu_arch             = Column("gpu_arch")
gpu_display_active   = Column("gpu_display_active")
gpu_frac             = Column("gpu_frac")
gpu_ids              = Column("gpu_ids")
gpu_lanes            = Column("gpu_lanes")
gpu_mem_bw           = Column("gpu_mem_bw")
gpu_name             = Column("gpu_name")
gpu_ram              = Column("gpu_ram")
gpu_total_ram        = Column("gpu_total_ram")
gpu_max_power        = Column("gpu_max_power")
gpu_max_temp         = Column("gpu_max_temp")
has_avx              = Column("has_avx")
host_id              = Column("host_id")
hosting_type         = Column("hosting_type")
hostname             = Column("hostname")
inet_down            = Column("inet_down")
inet_down_cost       = Column("inet_down_cost")
inet_up              = Column("inet_up")
inet_up_cost         = Column("inet_up_cost")
is_bid               = Column("is_bid")
logo                 = Column("logo")
machine_id           = Column("machine_id")
min_bid              = Column("min_bid")
mobo_name            = Column("mobo_name")
num_gpus             = Column("num_gpus")
os_version           = Column("os_version")
pci_gen              = Column("pci_gen")
pcie_bw              = Column("pcie_bw")
public_ipaddr        = Column("public_ipaddr")
reliability          = Column("reliability")
reliability_mult     = Column("reliability_mult")
rentable             = Column("rentable")
rented               = Column("rented")
score                = Column("score")
start_date           = Column("start_date")
static_ip            = Column("static_ip")
storage_cost         = Column("storage_cost")
storage_total_cost   = Column("storage_total_cost")
total_flops          = Column("total_flops")
verification         = Column("verification")
vericode             = Column("vericode")
vram_costperhour     = Column("vram_costperhour")
webpage              = Column("webpage")
vms_enabled          = Column("vms_enabled")
expected_reliability = Column("expected_reliability")
is_vm_deverified     = Column("is_vm_deverified")
resource_type        = Column("resource_type")
cluster_id           = Column("cluster_id")
avail_vol_ask_id     = Column("avail_vol_ask_id")
avail_vol_dph        = Column("avail_vol_dph")
avail_vol_size       = Column("avail_vol_size")
nw_disk_min_bw       = Column("nw_disk_min_bw")
nw_disk_max_bw       = Column("nw_disk_max_bw")
nw_disk_avg_bw       = Column("nw_disk_avg_bw")
platform_fee         = Column("platform_fee")

# Virtual columns accepted by the API that are converted to real column(s) by search_asks_
# (see vast/web/views/misc.py)
geolocation          = Column("geolocation")       # hashed → geolocode
datacenter           = Column("datacenter")        # bool → hosting_type {eq: 0|1}
duration             = Column("duration")          # seconds from now → end_date
verified             = Column("verified")          # bool → vericode
allocated_storage    = Column("allocated_storage") # GB → disk_space {gte: value}
target_reliability   = Column("target_reliability")# float → expected_reliability {gte: value}

# Column aliases (remapped to real columns before querying)
reliability2         = Column("reliability2")      # → reliability
dphtotal             = Column("dphtotal")          # → dph_total
gpu_totalram         = Column("gpu_totalram")      # → gpu_total_ram
gpu_totalram_GB      = Column("gpu_totalram_GB")   # → gpu_total_ram (value * 1000)
cpu_ram_GB           = Column("cpu_ram_GB")        # → cpu_ram (value * 1000)
gpu_ram_GB           = Column("gpu_ram_GB")        # → gpu_ram (value * 1000)
ubuntu_version       = Column("ubuntu_version")    # → os_version
driver_version       = Column("driver_version")    # → driver_vers
