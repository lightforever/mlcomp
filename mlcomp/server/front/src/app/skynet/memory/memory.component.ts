import {Component} from '@angular/core';
import {Paginator} from "../../paginator";
import {Location} from "@angular/common";
import {MemoryService} from "./memory.service";
import {Memory, MemoryFilter} from "../../models";
import {MatDialog} from "@angular/material/dialog";
import {MemoryAddDialogComponent} from "./memory-add-dialog";
import {Helpers} from "../../helpers";

@Component({
    selector: 'app-memory',
    templateUrl: './memory.component.html',
    styleUrls: ['./memory.component.css']
})
export class MemoryComponent extends Paginator<Memory> {
    displayed_columns: string[] = [
        'model',
        'variant',
        'num_classes',
        'batch_size',
        'img_size',
        'memory'
    ];

    model: string;
    variant: string;

    filter_hidden: boolean = true;
    filter_applied_text: string = '';
    selected: Memory;

    constructor(
        protected service: MemoryService,
        protected location: Location,
        public dialog: MatDialog
    ) {
        super(service, location);
    }

    onchange() {
        this.change.emit();
        let count = 0;
        if (this.model) count += 1;
        if (this.variant) count += 1;
        this.filter_applied_text = count > 0 ? `(${count} applied)` : '';
    }

    get_filter(): any {
        let res = new MemoryFilter();
        res.paginator = super.get_filter();
        res.model = this.model;
        res.variant = this.variant;

        return res;
    }

    add() {
        const dialogRef = this.dialog.open(MemoryAddDialogComponent, {
            width: '600px', height: '700px',
            data: {'name': ''}
        });

        dialogRef.afterClosed().subscribe(result => {
            if (result) {
                this.service.add(result).subscribe(_ => {
                    this.change.emit();
                });

            }
        });
    }

    edit() {
        const dialogRef = this.dialog.open(MemoryAddDialogComponent, {
            width: '600px', height: '700px',
            data: Helpers.clone(this.selected)
        });

        dialogRef.afterClosed().subscribe(result => {
            if (result) {
                this.service.edit(result).subscribe(_ => {
                    this.change.emit();
                });

            }
        });
    }

    copy() {
        const dialogRef = this.dialog.open(MemoryAddDialogComponent, {
            width: '600px', height: '700px',
            data: Helpers.clone(this.selected)
        });

        dialogRef.afterClosed().subscribe(result => {
            if (result) {
                this.service.add(result).subscribe(_ => {
                    this.change.emit();
                });

            }
        });
    }

    remove() {
        this.service.remove(this.selected.id).subscribe(result=>{
            this.selected = null;
            this.change.emit();
        });
    }
}
