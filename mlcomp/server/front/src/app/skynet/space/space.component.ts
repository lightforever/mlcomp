import {Component, ViewChild} from '@angular/core';
import {Paginator} from "../../paginator";
import {Space, SpaceFilter} from "../../models";
import {Location} from "@angular/common";
import {MatDialog} from "@angular/material/dialog";
import {SpaceService} from "./space.service";
import {Helpers} from "../../helpers";
import {SpaceAddDialogComponent} from "./space-add-dialog";
import {MatTableDataSource} from "@angular/material/table";
import {MatPaginator} from "@angular/material/paginator";
import {MatSort} from "@angular/material/sort";
import {SpaceRunDialogComponent} from "./space-run-dialog";

@Component({
    selector: 'app-space',
    templateUrl: './space.component.html',
    styleUrls: ['./space.component.css']
})
export class SpaceComponent extends Paginator<Space> {
    displayed_columns: string[] = [
        'name',
        'created',
        'changed'
    ];

    filter_hidden: boolean = true;
    filter_applied_text: string = '';

    name: string;
    selected: Space;

    relation_dataSource: MatTableDataSource<Space> = new MatTableDataSource();
    relation_total: number = 0;
    @ViewChild('relation_paginator') relation_paginator: MatPaginator;
    @ViewChild('relation_sort') relation_sort: MatSort;
    relation_selected: Space;
    relation_filter_hidden: boolean = true;
    relation_filter_applied_text: string = '';
    relation_name: string;

    constructor(
        protected service: SpaceService,
        protected location: Location,
        public dialog: MatDialog
    ) {
        super(service, location);
        this.id_column = 'name';
    }

    onchange() {
        this.change.emit();
        let count = 0;
        if (this.name) count += 1;
        this.filter_applied_text = count > 0 ? `(${count} applied)` : '';
    }

    get_filter(): any {
        let res = new SpaceFilter();
        res.paginator = super.get_filter();
        res.name = this.name;
        return res;
    }

    protected _ngOnInit() {
        super._ngOnInit();
        this.relation_paginator.page.subscribe(x => {
            this.relation_changed()
        });
        this.relation_sort.sortChange.subscribe(x => {
            this.relation_changed()
        });
    }


    run() {
        this.dialog.open(SpaceRunDialogComponent, {
            width: '1900px', height: '1050px',
            data: {'space': this.selected.name}
        });
    }

    add() {
        const dialogRef = this.dialog.open(SpaceAddDialogComponent, {
            width: '600px', height: '700px',
            data: {method: 'add', space: {'name': ''}}
        });

        dialogRef.afterClosed().subscribe(result => {
            this.change.emit();
        });
    }

    edit() {
        const dialogRef = this.dialog.open(SpaceAddDialogComponent, {
            width: '600px', height: '700px',
            data: {method: 'edit', space: Helpers.clone(this.selected)}
        });

        dialogRef.afterClosed().subscribe(result => {
            this.change.emit();
        });
    }

    copy() {
        const dialogRef = this.dialog.open(SpaceAddDialogComponent, {
            width: '600px', height: '700px',
            data: {
                method: 'copy',
                space: Helpers.clone(this.selected),
                'old_space': this.selected.name
            }
        });

        dialogRef.afterClosed().subscribe(result => {
            this.change.emit();
        });
    }

    remove() {
        this.service.remove(this.selected.name).subscribe(result => {
            this.selected = null;
            this.relation_selected = null;
            this.relation_dataSource.data = [];
            this.relation_paginator.pageIndex = 0;
            this.change.emit();
        });
    }

    relation_changed() {
        let filter = {
            'parent': this.selected.name,
            'paginator': {
                'page_number': this.relation_paginator.pageIndex,
                'page_size': this.relation_paginator.pageSize,
                'sort_column': this.relation_sort.active ? this.relation_sort.active : '',
                'sort_descending': this.relation_sort.direction ?
                    this.relation_sort.direction == 'desc' : true
            },
            'name': this.relation_name,
        };
        this.service.get_paginator<Space>(filter).subscribe(res => {
            this.relation_dataSource.data = res.data;
            this.relation_total = res.total;
        })
    }

    onSelected() {
        this.relation_selected = null;
        this.relation_paginator.pageIndex = 0;
        this.relation_changed();
    }

    relation_append() {
        this.service.relation_append(this.selected.name, this.relation_selected.name).subscribe(res => {
            this.relation_changed()
        });
    }

    relation_remove() {
        this.service.relation_remove(this.selected.name, this.relation_selected.name).subscribe(res => {
            this.relation_changed()
        });
    }
}
