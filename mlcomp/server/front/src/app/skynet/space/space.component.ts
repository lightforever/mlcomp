import {Component, ViewChild} from '@angular/core';
import {Paginator} from "../../paginator";
import {Dag, Space, SpaceFilter} from "../../models";
import {Location} from "@angular/common";
import {MatDialog} from "@angular/material/dialog";
import {SpaceService} from "./space.service";
import {Helpers} from "../../helpers";
import {SpaceAddDialogComponent} from "./space-add-dialog";
import {MatTableDataSource} from "@angular/material/table";
import {MatPaginator} from "@angular/material/paginator";
import {MatSort} from "@angular/material/sort";
import {SpaceRunDialogComponent} from "./space-run-dialog";
import {COMMA, ENTER} from "@angular/cdk/keycodes";
import {MatChipInputEvent} from "@angular/material/chips";
import {MatAutocompleteSelectedEvent} from "@angular/material/autocomplete";
import {MatIconRegistry} from "@angular/material/icon";
import {DomSanitizer} from "@angular/platform-browser";

@Component({
    selector: 'app-space',
    templateUrl: './space.component.html',
    styleUrls: ['./space.component.css']
})
export class SpaceComponent extends Paginator<Space> {
    displayed_columns: string[] = [
        'name',
        'created',
        'changed',
        'tags'
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

    separatorKeysCodes: number[] = [ENTER, COMMA];
    tags: string[] = [];


    constructor(
        protected service: SpaceService,
        protected location: Location,
        public dialog: MatDialog,
        iconRegistry: MatIconRegistry,
        sanitizer: DomSanitizer
    ) {
        super(service, location);
        this.id_column = 'name';

        iconRegistry.addSvgIcon('delete',
            sanitizer.bypassSecurityTrustResourceUrl(
                'assets/img/delete.svg'));
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

        this.data_updated.subscribe(res => {
            if (!res) {
                return;
            }
            this.tags = res.tags;
        });
    }


    run() {
        this.dialog.open(SpaceRunDialogComponent, {
            width: '2000px', height: '900px',
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

    remove_tag(space: Space, tag: string) {
        space.tags.splice(space.tags.indexOf(tag, 1));
        this.service.tag_remove(space.name, tag).subscribe(res => {
        });
        this.relation_changed();
    }

    tag_add(space: Space, event: MatChipInputEvent) {
        const input = event.input;
        let value = event.value;

        // Add our fruit
        if ((value || '').trim()) {
            value = value.trim();
            space.tags.push(value);
            this.service.tag_add(space.name, value).subscribe(res => {
            });
        }

        // Reset the input value
        if (input) {
            input.value = '';
        }

        this.relation_changed();
    }

    tag_selected(space: Space, event: MatAutocompleteSelectedEvent) {
        this.service.tag_add(space.name, event.option.viewValue).subscribe(res => {
        });
        space.tags.push(event.option.viewValue);
        this.relation_changed();
    }
}
